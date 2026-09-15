use super::*;
use ipu_codegen::{RepeatPointer, RepeatStep};
use ipu_target::ipu21::instruction::{encode_delay_m, encode_put_special_m, encode_setzi_m};

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
                ipu_exchange::paired_multicast(&topology, 0, &destinations, items)?
            } else if destinations.len() == 1 {
                let point = ipu_exchange::point_to_point(&topology, 0, destinations[0], words)?;
                MulticastPlan {
                    sender: point.sender,
                    receivers: vec![finalize_point_receiver(
                        &point.receiver,
                        topology.physical(0)?,
                    )?],
                }
            } else {
                ipu_exchange::multicast(&topology, 0, &destinations, words, 0)?
            };
            patch_sender_address(&mut plan.sender, if relative { 0 } else { source_address })?;
            for row in &mut plan.receivers {
                patch_receiver_address(row, destination_address)?;
            }
            let prepared = plan.prepare(0)?;
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
                    .map(|row| row.words().len() as u32 * 4)
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
                    .filter_map(|(tile, row)| {
                        row.as_ref().map(|row| (tile as u16, row.words().to_vec()))
                    })
                    .collect(),
            });
            exchanges.push((row_address, phase));
            row_address = (end + 7) & !7;
        }
        // Exercise moving -> absolute -> moving sources without another sync.
        let mut sections = Vec::new();
        for section in 0..3u32 {
            let relative = section != 1;
            let address = 0x6d000 + case * 3 * stride + section * stride;
            let mut plan = if paired {
                ipu_exchange::paired_multicast(&topology, 0, &destinations, words / 2)?
            } else if destinations.len() == 1 {
                let point = ipu_exchange::point_to_point(&topology, 0, destinations[0], words)?;
                MulticastPlan {
                    sender: point.sender,
                    receivers: vec![finalize_point_receiver(
                        &point.receiver,
                        topology.physical(0)?,
                    )?],
                }
            } else {
                ipu_exchange::multicast(&topology, 0, &destinations, words, 0)?
            };
            patch_sender_address(&mut plan.sender, if relative { 0 } else { source_address })?;
            if destinations.len() != 1 {
                for row in &mut plan.receivers {
                    patch_receiver_address(row, address)?;
                }
            }
            let prepared = plan.prepare(0)?;
            let helpers = if paired {
                vec![topology.paired_logical(0)?]
            } else {
                vec![]
            };
            let items = if paired { words / 2 } else { words };
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
            sections.push(builder.finish()?);
            for &tile in &destinations {
                data.push(TileProgramData {
                    tile,
                    address,
                    data: vec![0; stride as usize],
                });
                readbacks.push(ExpectedSpan {
                    tile,
                    address,
                    words: payload[if relative {
                        2 * words as usize..3 * words as usize
                    } else {
                        0..words as usize
                    }]
                    .to_vec(),
                });
            }
        }
        let grouped_address = row_address;
        let mut grouped_rows = Vec::new();
        for tile in 0..tiles {
            // The package prologue leaves the moving base in m6.
            let mut row = vec![encode_delay_m(1)?, encode_delay_m(1)?];
            for (section, phase) in sections.iter().enumerate() {
                if tile == 0 || destinations.len() == 1 {
                    row.extend([
                        encode_setzi_m(
                            8,
                            if destinations.len() == 1 {
                                0x6d000 + case * 3 * stride + section as u32 * stride
                            } else {
                                0
                            },
                        )?,
                        encode_put_special_m(0xa4, 8)?,
                        encode_put_special_m(0xa7, if section == 1 || tile != 0 { 15 } else { 6 })?,
                        encode_delay_m(1)?,
                    ]);
                } else {
                    // SETZI + two base writes must match exactly 27 timed
                    // cycles, not the ordinary scalar issue-rate estimate.
                    row.extend([
                        encode_delay_m(1)?,
                        encode_delay_m(1)?,
                        encode_delay_m(25)?,
                        encode_delay_m(1)?,
                    ]);
                }
                let mut body = phase.programs[usize::from(tile)]
                    .clone()
                    .map(ipu_exchange::EncodedRow::into_words)
                    .unwrap_or_else(inactive_exchange_program);
                assert_eq!(
                    body.pop(),
                    Some(ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION)
                );
                row.extend(body);
                // Equal timed sections on all tiles; retain eight-byte alignment.
                let padding = phase.event_cycles + 16 - phase.tile_event_cycles[usize::from(tile)];
                if row.len() % 2 == 0 {
                    row.push(encode_delay_m(1)?);
                    row.push(encode_delay_m(padding - 1)?);
                } else {
                    row.push(encode_delay_m(padding)?);
                }
            }
            row.push(ipu_target::ipu21::instruction::RETURN_M10_INSTRUCTION);
            row_address = row_address.max(grouped_address + row.len() as u32 * 4);
            grouped_rows.push((tile, row.clone()));
            let mut body = vec![TileStep::Exchange(ExchangeStep {
                outgoing_base: (tile == 0).then_some(TileAddress::RepeatPointer {
                    index: 0,
                    offset: 0,
                }),
                ..ExchangeStep::new(
                    true,
                    0,
                    PlacedExchangeRow {
                        address: grouped_address,
                        words: row,
                    },
                )
            })];
            if destinations.contains(&tile) {
                for section in 0..3u32 {
                    let address = 0x6d000 + case * 3 * stride + section * stride;
                    body.push(TileStep::Compute(ComputeStep {
                        symbol: "static_assert_equal_u32".into(),
                        output_address: TileAddress::Absolute(address),
                        input_addresses: vec![
                            TileAddress::Absolute(address),
                            if section == 1 {
                                TileAddress::Absolute(source_address)
                            } else {
                                TileAddress::RepeatPointer {
                                    index: 0,
                                    offset: 0,
                                }
                            },
                        ],
                        arguments: vec![words],
                        profile: StepProfile::default(),
                    }));
                }
            }
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
        }
        diagnostic_rows.push(StressRow {
            case: Some(case),
            address: grouped_address,
            end: row_address,
            programs: grouped_rows.into_iter().collect(),
        });
        row_address = (row_address + 7) & !7;
        for tile in 0..tiles {
            for (relative, (address, phase)) in [true, false].into_iter().zip(&exchanges) {
                let mut body = vec![TileStep::Exchange(ExchangeStep {
                    outgoing_base: (relative && tile == 0).then_some(TileAddress::RepeatPointer {
                        index: 0,
                        offset: 0,
                    }),
                    ..ExchangeStep::new(
                        phase.programs[usize::from(tile)].is_some(),
                        0,
                        PlacedExchangeRow {
                            address: *address,
                            words: phase.programs[usize::from(tile)]
                                .clone()
                                .map(ipu_exchange::EncodedRow::into_words)
                                .unwrap_or_else(inactive_exchange_program),
                        },
                    )
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
