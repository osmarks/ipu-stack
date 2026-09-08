use super::*;

/// Test whether the DELTA CSRs are strides or mutable address cursors. Record
/// them immediately around the exchange, before host readback changes state.
pub(crate) fn build(toolchain: &Toolchain, runtime_source: &Path) -> Result<StressPackage> {
    use ipu_exchange::{encode_put_special_m, encode_setzi_m, encode_st32_m_immediate};
    let topology = Topology::c600();
    let tiles = u16::try_from(topology.tile_count())?;
    let words = 16;
    let mut plan = topology.multicast(0, &[1, 2], words, 0)?;
    patch_sender_address(&mut plan.sender, SOURCE_BASE)?;
    for row in &mut plan.receivers {
        patch_receiver_address(row, DATA_BASE)?;
    }
    let plan = plan.prepare()?;
    let mut builder = PhaseProgramBuilder::new(tiles);
    let offset = builder.earliest_transfer_offset(0, &[], &[1, 2], &plan, words, 0)?;
    builder.append_transfer_at(0, &[], &[1, 2], &plan, offset, words)?;
    let phase = builder.finish()?;
    let snapshot = 0x70000;
    let mut programs = Vec::new();
    let mut data = Vec::new();
    let mut readbacks = Vec::new();
    let mut rows = Vec::new();
    for tile in 0..tiles {
        let mut row = phase.programs[usize::from(tile)]
            .clone()
            .unwrap_or_else(inactive_exchange_program);
        // Four instructions preserve the row's eight-byte instruction alignment.
        let mut prefix = vec![
            encode_setzi_m(0, 8)?,
            encode_put_special_m(0xa2, 0)?,
            encode_put_special_m(0xa8, 0)?,
            encode_setzi_m(1, snapshot)?,
        ];
        for (index, csr) in [0xa2, 0xa8].into_iter().enumerate() {
            prefix.push(0x4100_0000 | csr); // get m0, CSR (also used by ipu-driver)
            prefix.push(encode_st32_m_immediate(0, 1, 15, index as u16)?);
        }
        // All tiles execute the same preamble after the global barrier.
        prefix.insert(0, ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
        prefix.push(encode_exchange_delay(0));
        prefix.append(&mut row);
        row = prefix;
        assert_eq!(row.pop(), Some(ipu_exchange::RETURN_M10_INSTRUCTION));
        for (index, csr) in [0xa2, 0xa8].into_iter().enumerate() {
            row.push(0x4100_0000 | csr);
            row.push(encode_st32_m_immediate(0, 1, 15, index as u16 + 2)?);
        }
        row.push(ipu_exchange::RETURN_M10_INSTRUCTION);
        rows.push((tile, row.clone()));
        programs.push(TileProgram {
            tile,
            steps: vec![TileStep::Exchange(ExchangeStep {
                active: true,
                incoming_base: 0,
                preserve_base_registers: false,
                incoming_mux: None,
                incoming_format: 0,
                incoming_mux_pair: None,
                incoming_dcount: None,
                sync_in_program: true,
                program: PlacedExchangeRow {
                    address: ROW_BASE,
                    words: row,
                },
                setup_patch: None,
                repeat_patches: Vec::new(),
                profile: StepProfile::default(),
            })],
        });
        if tile < 3 {
            let payload = (0..256u32).map(|i| 0x12340000 + i).collect::<Vec<_>>();
            data.push(TileProgramData {
                tile,
                address: SOURCE_BASE,
                data: payload.iter().flat_map(|v| v.to_le_bytes()).collect(),
            });
            data.push(TileProgramData {
                tile,
                address: DATA_BASE,
                data: vec![0; 1024],
            });
            if tile != 0 {
                let expected = payload[..words as usize].to_vec();
                readbacks.push(ExpectedSpan {
                    tile,
                    address: DATA_BASE,
                    words: expected,
                });
            }
            data.push(TileProgramData {
                tile,
                address: snapshot,
                data: vec![0; 16],
            });
            // These predictions deliberately distinguish the cursor hypothesis
            // from a stride: initialized deltas get overwritten by PIC/SEND.
            let state = ExpectedSpan {
                tile,
                address: snapshot,
                words: vec![
                    8,
                    8,
                    if tile == 0 { 8 } else { DATA_BASE + words * 4 },
                    if tile == 0 {
                        SOURCE_BASE + words * 4
                    } else {
                        8
                    },
                ],
            };
            readbacks.push(state);
        }
    }
    let outputs = readback_bindings(&readbacks, &topology)?;
    let application =
        build_tile_program_package(&programs, &data, &outputs, toolchain, runtime_source)?;
    let end = ROW_BASE
        + rows
            .iter()
            .map(|(_, row)| row.len() as u32 * 4)
            .max()
            .unwrap();
    Ok(StressPackage {
        application,
        active_tiles: 3,
        transfers: Vec::new(),
        rows: vec![StressRow {
            case: Some(0),
            address: ROW_BASE,
            end,
            programs: rows.into_iter().collect(),
        }],
        readbacks,
    })
}
