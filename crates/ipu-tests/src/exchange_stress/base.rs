use super::*;
use ipu_codegen::{RepeatPointer, RepeatStep};

/// Exercise exact payloads on every Repeat iteration, then an absolute row to
/// check that a subsequent exchange resets OUTGOING_BASE.
pub(crate) fn build(toolchain: &Toolchain, runtime_source: &Path) -> Result<StressPackage> {
    let topology = Topology::c600();
    let tiles = u16::try_from(topology.tile_count())?;
    let words = 128u32;
    let stride = words * 4;
    let mut programs = (0..tiles)
        .map(|tile| TileProgram {
            tile,
            steps: vec![],
        })
        .collect::<Vec<_>>();
    let mut data = Vec::new();
    let mut readbacks = Vec::new();
    let mut diagnostic_rows = Vec::new();
    let mut row_address = ROW_BASE;
    for case in 0..8u32 {
        let paired = case % 4 == 2;
        let destinations = match case % 4 {
            0 => vec![2],
            1 | 2 => vec![2, 3],
            _ => vec![0, 2],
        };
        let source_address = if case < 4 {
            SOURCE_BASE
        } else {
            INTERLEAVED_SOURCE_BASE
        };
        let source_address = source_address + (case % 4) * 3 * stride;
        let destination_address = DATA_BASE + case * stride;
        let payload = (0..3 * words)
            .map(|i| 0xa531_0000 ^ (case << 16) ^ i.wrapping_mul(0x9e37_79b9))
            .collect::<Vec<_>>();
        for tile in std::iter::once(0)
            .chain(destinations.iter().copied())
            .collect::<BTreeSet<_>>()
        {
            data.push(TileProgramData {
                tile,
                address: source_address,
                data: payload.iter().flat_map(|w| w.to_le_bytes()).collect(),
            });
        }
        for &tile in &destinations {
            data.push(TileProgramData {
                tile,
                address: destination_address,
                data: vec![0; stride as usize],
            });
            readbacks.push(ExpectedSpan {
                tile,
                address: destination_address,
                words: payload[..words as usize].to_vec(),
            });
        }
        let mut exchanges = Vec::new();
        for relative in [true, false] {
            let items = if paired { words / 2 } else { words };
            let mut plan = if paired {
                topology.paired_multicast(0, &destinations, items)?
            } else if destinations.len() == 1 {
                let point = topology.point_to_point(0, destinations[0], words)?;
                MulticastPlan {
                    sender: point.sender,
                    receivers: vec![finalize_point_receiver(
                        &point.receiver,
                        topology.physical(0)?,
                    )?],
                }
            } else {
                topology.multicast(0, &destinations, words, 0)?
            };
            patch_sender_address(&mut plan.sender, if relative { 0 } else { source_address })?;
            for row in &mut plan.receivers {
                patch_receiver_address(row, destination_address)?;
            }
            let prepared = plan.prepare()?;
            let helpers = if paired {
                vec![topology.paired_logical(0)?]
            } else {
                vec![]
            };
            let mut builder = PhaseProgramBuilder::new(tiles);
            let offset = builder.earliest_transfer_offset(
                0,
                &helpers,
                &destinations,
                &prepared,
                items,
                0,
            )?;
            builder.append_transfer_at(0, &helpers, &destinations, &prepared, offset, items)?;
            let phase = builder.finish()?;
            let end = row_address
                + phase
                    .programs
                    .iter()
                    .flatten()
                    .map(|row| row.len() as u32 * 4)
                    .max()
                    .unwrap();
            diagnostic_rows.push(StressRow {
                case: Some(case),
                address: row_address,
                end,
                programs: phase
                    .programs
                    .iter()
                    .enumerate()
                    .filter_map(|(tile, row)| row.clone().map(|row| (tile as u16, row)))
                    .collect(),
            });
            exchanges.push((row_address, phase));
            row_address = (end + 7) & !7;
        }
        for tile in 0..tiles {
            for (relative, (address, phase)) in [true, false].into_iter().zip(&exchanges) {
                let mut body = vec![TileStep::Exchange(ExchangeStep {
                    active: phase.programs[usize::from(tile)].is_some(),
                    incoming_base: 0,
                    outgoing_base: (relative && tile == 0).then_some(TileAddress::RepeatPointer {
                        index: 0,
                        offset: 0,
                    }),
                    preserve_base_registers: false,
                    incoming_mux: None,
                    incoming_format: 0,
                    incoming_mux_pair: None,
                    incoming_dcount: None,
                    sync_in_program: false,
                    program: PlacedExchangeRow {
                        address: *address,
                        words: phase.programs[usize::from(tile)]
                            .clone()
                            .unwrap_or_else(inactive_exchange_program),
                    },
                    setup_patch: None,
                    repeat_patches: vec![],
                    profile: StepProfile::default(),
                })];
                if destinations.contains(&tile) {
                    body.push(TileStep::Compute(ComputeStep {
                        symbol: "static_assert_equal_u32".into(),
                        output_address: TileAddress::Absolute(destination_address),
                        input_addresses: vec![
                            TileAddress::Absolute(destination_address),
                            if relative {
                                TileAddress::RepeatPointer {
                                    index: 0,
                                    offset: 0,
                                }
                            } else {
                                TileAddress::Absolute(source_address)
                            },
                        ],
                        arguments: vec![words],
                        profile: StepProfile::default(),
                    }));
                }
                if relative {
                    programs[usize::from(tile)]
                        .steps
                        .push(TileStep::Repeat(RepeatStep {
                            count: 3,
                            iterated_pointers: vec![RepeatPointer {
                                initial_address: source_address,
                                stride_bytes: stride,
                            }],
                            body,
                            profile: StepProfile::default(),
                        }));
                } else {
                    programs[usize::from(tile)].steps.extend(body);
                }
            }
        }
    }
    let outputs = readback_bindings(&readbacks, &topology)?;
    let application =
        build_tile_program_package(&programs, &data, &outputs, toolchain, runtime_source)?;
    Ok(StressPackage {
        application,
        active_tiles: 4,
        transfers: vec![],
        rows: diagnostic_rows,
        readbacks,
    })
}
