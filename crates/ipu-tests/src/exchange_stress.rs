use anyhow::{Context, Result, bail};
use ipu_codegen::{
    CheckpointStep, CompiledPackage, ComputeStep, ExchangeActivity, ExchangeActivityKind,
    ExchangeStep, PlacedExchangeRow, StepProfile, TileAddress, TileProgram, TileProgramData,
    TileStep, build_tile_program_package, inactive_exchange_program,
};
use ipu_driver::{Device, TileException};
use ipu_elf::Toolchain;
use ipu_exchange::{
    MulticastPlan, PhaseProgramBuilder, PhaseTransferTiming, Topology, encode_exchange_delay,
    finalize_point_receiver, patch_receiver_address, patch_sender_address,
    scheduled_receiver_timing,
};
use ipu_package::{Application, Binding, RegionSlice};
use ipu_runtime::Runtime;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const DATA_BASE: u32 = 0x60000;
const SOURCE_BASE: u32 = 0x65000;
const WIDE_DESTINATION_BASE: u32 = DATA_BASE;
const EXPECTED_BASE: u32 = 0x6c000;
const DATA_LIMIT: u32 = 0x73800;
const ROW_BASE: u32 = 0x5c000;
const WIDE_ROW_BASE: u32 = 0x5c000;
const WIDE_ROW_LIMIT: u32 = 0x60000;
const INTERLEAVED_SOURCE_BASE: u32 = 0x88000;
const INTERLEAVED_DESTINATION_BASE: u32 = 0x98000;

#[derive(Clone, Debug)]
struct Transfer {
    case: u32,
    source: u16,
    destinations: Vec<u16>,
    source_address: u32,
    destination_addresses: Vec<u32>,
    words: u32,
    requested_schedule_offset: u32,
    schedule_offset: u32,
    timing: PhaseTransferTiming,
}

#[derive(Clone, Debug)]
struct TransferSpec {
    source: u16,
    destinations: Vec<u16>,
    words: u32,
    schedule_offset: Option<u32>,
}

#[derive(Clone, Debug)]
struct Payload {
    case: u32,
    source: u16,
    destinations: Vec<u16>,
    words: u32,
}

pub(crate) struct StressPackage {
    pub application: Application,
    active_tiles: u16,
    transfers: Vec<Transfer>,
    rows: Vec<StressRow>,
    readbacks: Vec<ExpectedSpan>,
}

pub(crate) struct PhaseReplayPackage {
    pub application: Application,
    pub phase: usize,
    expected: Vec<ExpectedSpan>,
    activities: Vec<Vec<ExchangeActivity>>,
    initial_origins: BTreeMap<u32, Vec<(u16, u32)>>,
}

pub(crate) fn build_wide(
    active_tiles: u16,
    first_case: u32,
    cases: u32,
    words: u32,
    validate: bool,
    receiver_mask: u8,
    explicit_config: bool,
    all_active: bool,
    receiver_pairs: u16,
    source: u16,
    first_destination: u16,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<StressPackage> {
    if active_tiles < 4 {
        bail!("the paired 64-bit exchange matrix requires at least four active tiles");
    }
    if cases == 0 || first_case >= 16 || first_case + cases > 16 {
        bail!("the paired 64-bit exchange matrix selects cases from 0..16");
    }
    if receiver_mask > 0b11 {
        bail!("paired 64-bit receiver mask must fit in two bits");
    }
    let destination_end = first_destination.saturating_add(receiver_pairs.saturating_mul(2));
    if receiver_pairs == 0
        || source >= active_tiles
        || (source ^ 1) >= active_tiles
        || first_destination & 1 != 0
        || destination_end > active_tiles
        || (first_destination..destination_end).contains(&source)
        || (first_destination..destination_end).contains(&(source ^ 1))
    {
        bail!("paired 64-bit diagnostic has invalid source or receiver-pair tiles");
    }
    let topology = Topology::c600();
    let execution_tiles = u16::try_from(topology.tile_count())?;
    if words < 128 || words & 1 != 0 || words / 2 > ipu_exchange::MAX_TRANSFER_WORDS {
        bail!("paired 64-bit exchange payload must contain 128..=8296 even u32 words");
    }
    let items = words / 2;
    let payload = (0..words)
        .map(|word| 0x6400_0000 ^ word.wrapping_mul(0x9e37_79b9))
        .collect::<Vec<_>>();
    let payload_bytes = payload
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let mut programs = (0..execution_tiles)
        .map(|tile| TileProgram {
            tile,
            steps: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut data = Vec::new();
    let mut transfers = Vec::new();
    let mut diagnostic_rows = Vec::new();
    let mut readbacks = Vec::new();
    let mut initialized = BTreeSet::new();
    let mut validated = BTreeSet::new();
    let mut row_address = WIDE_ROW_BASE;
    let setup_row = vec![
        ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION,
        encode_exchange_delay(0),
        ipu_exchange::RETURN_M10_INSTRUCTION,
    ];
    let setup_end = row_address + u32::try_from(setup_row.len())? * 4;
    diagnostic_rows.push(StressRow {
        case: None,
        address: row_address,
        end: setup_end,
        programs: (0..execution_tiles)
            .map(|tile| (tile, setup_row.clone()))
            .collect(),
    });
    for tile in 0..execution_tiles {
        programs[usize::from(tile)]
            .steps
            .push(TileStep::Exchange(ExchangeStep {
                active: true,
                incoming_base: 0,
                preserve_base_registers: false,
                incoming_mux: None,
                incoming_format: 0,
                incoming_mux_pair: None,
                incoming_dcount: None,
                sync_in_program: true,
                program: PlacedExchangeRow {
                    address: row_address,
                    words: setup_row.clone(),
                },
                setup_patch: None,
                repeat_patches: Vec::new(),
                profile: StepProfile::default(),
            }));
    }
    row_address = (setup_end + 7) & !7;
    let region_bases = [
        (SOURCE_BASE, WIDE_DESTINATION_BASE, "standard->standard"),
        (
            INTERLEAVED_SOURCE_BASE,
            INTERLEAVED_DESTINATION_BASE,
            "interleaved->interleaved",
        ),
        (
            SOURCE_BASE,
            INTERLEAVED_DESTINATION_BASE,
            "standard->interleaved",
        ),
        (
            INTERLEAVED_SOURCE_BASE,
            WIDE_DESTINATION_BASE,
            "interleaved->standard",
        ),
    ];

    for (case, &(source_base, destination_base, region_name)) in region_bases
        .iter()
        .enumerate()
        .flat_map(|(region, bases)| (0..4).map(move |bank_case| (region * 4 + bank_case, bases)))
        .skip(usize::try_from(first_case)?)
        .take(usize::try_from(cases)?)
    {
        let destinations = (first_destination..destination_end).collect::<Vec<_>>();
        let bank_case = case & 3;
        let source_element_size = if source_base >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE {
            ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE
        } else {
            ipu_package::TILE_MEMORY_ELEMENT_SIZE
        };
        let destination_element_size =
            if destination_base >= ipu_package::IPU21_INTERLEAVED_MEMORY_BASE {
                ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_package::TILE_MEMORY_ELEMENT_SIZE
            };
        let source_bank_offset = (u32::try_from(bank_case)? >> 1) * source_element_size;
        let destination_bank_offset = (u32::try_from(bank_case)? & 1) * destination_element_size;
        let payload_stride = (words * 4 + 7) & !7;
        let region_slot = u32::try_from(case / 8)?;
        if (region_slot + 1) * payload_stride > source_element_size.min(destination_element_size) {
            bail!(
                "paired exchange bank matrix payloads do not fit within their selected memory elements"
            );
        }
        let region_offset = region_slot * payload_stride;
        let source_address = source_base + region_offset + source_bank_offset;
        let destination_address = destination_base + region_offset + destination_bank_offset;

        let mut plan = topology.paired_multicast(source, &destinations, items)?;
        patch_sender_address(&mut plan.sender, source_address)?;
        for row in &mut plan.receivers {
            patch_receiver_address(row, destination_address)?;
        }
        let mut builder = PhaseProgramBuilder::new(execution_tiles);
        let paired_source = topology.paired_logical(source)?;
        let schedule_offset = builder.earliest_transfer_offset(
            source,
            &[paired_source],
            &destinations,
            &plan,
            items,
            0,
        )?;
        let timing = builder.append_transfer_at(
            source,
            &[paired_source],
            &destinations,
            &plan,
            schedule_offset,
            items,
        )?;
        let phase = builder.finish()?;
        let mut rows = phase.programs;
        for row in rows.iter_mut().flatten() {
            row.insert(0, ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION);
        }
        for (index, &destination) in destinations.iter().enumerate() {
            if receiver_mask & (1 << (index & 1)) == 0 {
                rows[usize::from(destination)] = None;
            }
        }
        if all_active {
            for row in &mut rows {
                if row.is_none() {
                    *row = Some(vec![
                        ipu_exchange::SYNC_SUPERVISOR_INSTRUCTION,
                        encode_exchange_delay(0),
                        ipu_exchange::RETURN_M10_INSTRUCTION,
                    ]);
                }
            }
        }
        let row_words = rows
            .iter()
            .filter_map(|row| row.as_ref().map(Vec::len))
            .max()
            .unwrap_or(1);
        let row_end = row_address + u32::try_from(row_words)? * 4;
        if row_end > WIDE_ROW_LIMIT {
            bail!("paired exchange rows exceed the diagnostic row region");
        }
        diagnostic_rows.push(StressRow {
            case: Some(u32::try_from(case)?),
            address: row_address,
            end: row_end,
            programs: rows
                .iter()
                .enumerate()
                .filter_map(|(tile, row)| {
                    row.clone()
                        .map(|words| (u16::try_from(tile).expect("tile index fits u16"), words))
                })
                .collect(),
        });

        for tile in 0..execution_tiles {
            let row = rows[usize::from(tile)]
                .clone()
                .unwrap_or_else(inactive_exchange_program);
            let active = rows[usize::from(tile)].is_some();
            let receiving = destinations.contains(&tile) && active;
            programs[usize::from(tile)]
                .steps
                .push(TileStep::Exchange(ExchangeStep {
                    active,
                    incoming_base: 0,
                    preserve_base_registers: true,
                    incoming_mux: None,
                    incoming_format: if receiving && explicit_config {
                        if topology.paired_receiver_is_early(tile, source)? {
                            1
                        } else {
                            2
                        }
                    } else {
                        0
                    },
                    incoming_mux_pair: (receiving && explicit_config)
                        .then_some(topology.paired_source_mux(source)?),
                    incoming_dcount: None,
                    sync_in_program: active,
                    program: PlacedExchangeRow {
                        address: row_address,
                        words: row,
                    },
                    setup_patch: None,
                    repeat_patches: Vec::new(),
                    profile: StepProfile::default(),
                }));
        }
        if initialized.insert((source, source_address)) {
            data.push(TileProgramData {
                tile: source,
                address: source_address,
                data: payload_bytes.clone(),
            });
        }
        if validate && validated.insert((source, source_address)) {
            readbacks.push(ExpectedSpan {
                tile: source,
                address: source_address,
                words: payload.clone(),
            });
        }
        for &tile in &destinations {
            if initialized.insert((tile, destination_address)) {
                data.push(TileProgramData {
                    tile,
                    address: destination_address,
                    data: vec![0; payload_bytes.len()],
                });
            }
            let receiver_index = destinations
                .iter()
                .position(|candidate| *candidate == tile)
                .expect("destination comes from receiver pair");
            if validate
                && receiver_mask & (1 << (receiver_index & 1)) != 0
                && validated.insert((tile, destination_address))
            {
                readbacks.push(ExpectedSpan {
                    tile,
                    address: destination_address,
                    words: payload.clone(),
                });
            }
        }
        transfers.push(Transfer {
            case: u32::try_from(case)?,
            source,
            destinations: destinations.to_vec(),
            source_address,
            destination_addresses: vec![destination_address; destinations.len()],
            words,
            requested_schedule_offset: 0,
            schedule_offset: 0,
            timing,
        });
        eprintln!(
            "exchangeWide case={case} region={region_name} sourceElement={} destinationElement={} source={source} destinations={destinations:?}",
            source_bank_offset / source_element_size,
            destination_bank_offset / destination_element_size,
        );
        row_address = (row_end + 7) & !7;
    }

    let output_bindings = readbacks
        .iter()
        .enumerate()
        .map(|(index, span)| -> Result<Binding> {
            Ok(Binding {
                name: format!("paired-result-{index}"),
                dtype: "u32".into(),
                shape: vec![u32::try_from(span.words.len())?],
                slices: vec![RegionSlice {
                    tile: u32::from(topology.physical(span.tile)?),
                    tile_address: span.address,
                    file_offset: 0,
                    size: u64::try_from(span.words.len() * 4)?,
                }],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let application = build_tile_program_package(
        &programs,
        &data,
        &output_bindings,
        toolchain,
        runtime_source,
    )?;
    Ok(StressPackage {
        application,
        active_tiles,
        transfers,
        rows: diagnostic_rows,
        readbacks,
    })
}

#[derive(Clone, Debug)]
struct ExpectedSpan {
    tile: u16,
    address: u32,
    words: Vec<u32>,
}

#[derive(Clone, Debug)]
struct ReplayTransfer {
    send: ExchangeActivity,
    receives: Vec<(u16, ExchangeActivity)>,
}

#[derive(Clone, Copy, Debug)]
enum ReplayEvent {
    Receive {
        transfer: u32,
        tile: u16,
        address: u32,
    },
    Send {
        transfer: u32,
        tile: u16,
        address: u32,
        words: u32,
    },
}

#[derive(Clone, Debug)]
struct StressRow {
    case: Option<u32>,
    address: u32,
    end: u32,
    programs: BTreeMap<u16, Vec<u32>>,
}

pub(crate) fn build(
    seed: u64,
    active_tiles: u16,
    cases: u32,
    maximum_words: u32,
    maximum_transfers: u32,
    maximum_compute_delay: u32,
    overlap_sweep: bool,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<StressPackage> {
    let maximum_tiles = Topology::c600().tile_count();
    if active_tiles < 2 || usize::from(active_tiles) > maximum_tiles {
        bail!("exchange stress requires 2..={maximum_tiles} active tiles");
    }
    if overlap_sweep && active_tiles < 3 {
        bail!("the exchange overlap sweep requires at least three active tiles");
    }
    if cases == 0 {
        bail!("--exchange-cases must be nonzero");
    }
    if maximum_words == 0 || maximum_words > ipu_exchange::MAX_TRANSFER_WORDS {
        bail!(
            "--exchange-max-words must be in 1..={}",
            ipu_exchange::MAX_TRANSFER_WORDS
        );
    }
    if maximum_transfers == 0 {
        bail!("--exchange-max-transfers must be nonzero");
    }
    if maximum_compute_delay == 0 {
        bail!("--exchange-compute-delay must be nonzero");
    }
    let topology = Topology::c600();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut destination_cursors = vec![DATA_BASE; usize::from(active_tiles)];
    let mut source_cursors = vec![SOURCE_BASE; usize::from(active_tiles)];
    let mut expected_cursors = vec![EXPECTED_BASE; usize::from(active_tiles)];
    let mut buffers = BTreeMap::<u16, Vec<u8>>::new();
    let mut transfers = Vec::new();
    let mut available_payloads = vec![Vec::<(u32, Payload)>::new(); usize::from(active_tiles)];
    let mut phase_rows = Vec::with_capacity(cases as usize);
    let mut diagnostic_rows = Vec::with_capacity(cases as usize);
    let mut row_address = ROW_BASE;
    let mut previous_shape: Option<Vec<TransferSpec>> = None;

    for case in 0..cases {
        let mut tiles = (0..active_tiles).collect::<Vec<_>>();
        rng.shuffle(&mut tiles);
        let group_tiles = if overlap_sweep {
            3
        } else {
            rng.usize(2..=8).min(tiles.len())
        };
        let group = &tiles[..group_tiles];
        let contiguous_receiver = (!overlap_sweep && case == 0).then_some(group[0]);
        let shape = if overlap_sweep {
            overlap_specs(&topology, case, &group[..3], maximum_words, &mut rng)?
        } else if let Some(receiver) = contiguous_receiver {
            let sources = group
                .iter()
                .copied()
                .filter(|tile| *tile != receiver)
                .collect::<Vec<_>>();
            let words = paired_control_words(&topology, sources[0], receiver, maximum_words)?
                .unwrap_or_else(|| random_words(&mut rng, maximum_words));
            (0..usize::try_from(maximum_transfers)?)
                .map(|index| {
                    if index & 1 == 0 {
                        TransferSpec {
                            source: sources[(index / 2) % sources.len()],
                            destinations: vec![receiver],
                            words,
                            schedule_offset: None,
                        }
                    } else {
                        TransferSpec {
                            source: receiver,
                            destinations: vec![sources[(index / 2 + 1) % sources.len()]],
                            words,
                            schedule_offset: None,
                        }
                    }
                })
                .collect::<Vec<_>>()
        } else if rng.bool() {
            previous_shape.clone().unwrap_or_default()
        } else {
            Vec::new()
        };
        let shape = if shape.is_empty() {
            let mut shape = Vec::new();
            for _ in 0..rng.usize(1..=usize::try_from(maximum_transfers)?) {
                let source = group[rng.usize(0..group.len())];
                let mut destinations = group
                    .iter()
                    .copied()
                    .filter(|tile| *tile != source)
                    .collect::<Vec<_>>();
                rng.shuffle(&mut destinations);
                destinations.truncate(rng.usize(1..=3).min(destinations.len()));
                shape.push(TransferSpec {
                    source,
                    destinations,
                    words: random_words(&mut rng, maximum_words),
                    schedule_offset: None,
                });
            }
            previous_shape = Some(shape.clone());
            shape
        } else {
            shape
        };
        let mut builder = PhaseProgramBuilder::new(u16::try_from(topology.tile_count())?);
        let mut validators = BTreeMap::<u16, Vec<(u32, u32, u32)>>::new();
        for TransferSpec {
            source,
            destinations,
            mut words,
            schedule_offset,
        } in shape
        {
            let chained = (contiguous_receiver.is_none()
                && !available_payloads[usize::from(source)].is_empty()
                && rng.usize(0..4) == 0)
                .then(|| {
                    let candidates = &available_payloads[usize::from(source)];
                    candidates[rng.usize(0..candidates.len())].clone()
                });
            let (source_address, payload) = if let Some((address, payload)) = chained {
                words = payload.words;
                (address, payload)
            } else {
                let payload = Payload {
                    case,
                    source,
                    destinations: destinations.clone(),
                    words,
                };
                let address = allocate(
                    &mut source_cursors,
                    source,
                    words * 4,
                    EXPECTED_BASE,
                    &mut rng,
                )?;
                fill_source(
                    buffers
                        .entry(source)
                        .or_insert_with(|| vec![0; (DATA_LIMIT - DATA_BASE) as usize]),
                    address,
                    &payload,
                );
                (address, payload)
            };
            let bytes = words * 4;
            let destination_addresses = destinations
                .iter()
                .map(|&tile| {
                    if contiguous_receiver == Some(tile) {
                        allocate_with_fixed_padding(
                            &mut destination_cursors,
                            tile,
                            bytes,
                            u32::try_from(std::mem::size_of::<u32>())?,
                            SOURCE_BASE,
                        )
                    } else {
                        allocate(&mut destination_cursors, tile, bytes, SOURCE_BASE, &mut rng)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let expected_addresses = destinations
                .iter()
                .map(|&tile| allocate(&mut expected_cursors, tile, bytes, DATA_LIMIT, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let mut plan = if destinations.len() == 1 {
                let point = topology.point_to_point(source, destinations[0], words)?;
                MulticastPlan {
                    sender: point.sender,
                    receivers: vec![finalize_point_receiver(
                        &point.receivers[0],
                        topology.physical(source)?,
                    )?],
                }
            } else {
                topology.multicast(source, &destinations, words, 0)?
            };
            patch_sender_address(&mut plan.sender, source_address)?;
            for (row, &address) in plan.receivers.iter_mut().zip(&destination_addresses) {
                patch_receiver_address(row, address)?;
            }
            let requested_schedule_offset = schedule_offset.unwrap_or(0);
            let schedule_offset = builder.earliest_transfer_offset(
                source,
                &[],
                &destinations,
                &plan,
                words,
                requested_schedule_offset,
            )?;
            let timing = builder
                .append_transfer_at(source, &[], &destinations, &plan, schedule_offset, words)
                .with_context(|| {
                    format!(
                        "case {case} cannot encode transfer {source} -> {destinations:?} at schedule offset {schedule_offset}"
                    )
                })?;
            for ((&tile, &destination), &expected) in destinations
                .iter()
                .zip(&destination_addresses)
                .zip(&expected_addresses)
            {
                fill_source(
                    buffers
                        .entry(tile)
                        .or_insert_with(|| vec![0; (DATA_LIMIT - DATA_BASE) as usize]),
                    expected,
                    &payload,
                );
                available_payloads[usize::from(tile)].push((destination, payload.clone()));
                validators
                    .entry(tile)
                    .or_default()
                    .push((destination, expected, words));
            }
            transfers.push(Transfer {
                case,
                source,
                destinations,
                source_address,
                destination_addresses,
                words,
                requested_schedule_offset,
                schedule_offset,
                timing,
            });
        }
        let rows = builder.finish()?.programs;
        let maximum_row_words = rows
            .iter()
            .filter_map(|row| row.as_ref().map(Vec::len))
            .max()
            .unwrap_or(1);
        let row_bytes = u32::try_from(maximum_row_words)?
            .checked_mul(4)
            .context("row size overflow")?;
        let row_end = row_address
            .checked_add(row_bytes)
            .context("row address overflow")?;
        if row_end > DATA_BASE {
            bail!("{cases} stress cases exceed the exchange-row test region");
        }
        diagnostic_rows.push(StressRow {
            case: Some(case),
            address: row_address,
            end: row_end,
            programs: rows
                .iter()
                .enumerate()
                .filter_map(|(tile, row)| {
                    row.clone()
                        .map(|program| (u16::try_from(tile).expect("tile count is u16"), program))
                })
                .collect(),
        });
        phase_rows.push((row_address, rows, validators));
        row_address = (row_end + 7) & !7;
    }

    let execution_tiles = u16::try_from(topology.tile_count())?;
    let mut programs = (0..execution_tiles)
        .map(|tile| TileProgram {
            tile,
            steps: Vec::with_capacity(cases as usize),
        })
        .collect::<Vec<_>>();
    for (address, rows, validators) in phase_rows {
        for tile in 0..execution_tiles {
            let row = rows[usize::from(tile)]
                .clone()
                .unwrap_or_else(inactive_exchange_program);
            let active = rows[usize::from(tile)].is_some();
            programs[usize::from(tile)]
                .steps
                .push(TileStep::Exchange(ExchangeStep {
                    active,
                    incoming_base: 0,
                    preserve_base_registers: false,
                    incoming_mux: None,
                    incoming_format: 0,
                    incoming_mux_pair: None,
                    incoming_dcount: None,
                    sync_in_program: false,
                    program: PlacedExchangeRow {
                        address,
                        words: row,
                    },
                    setup_patch: None,
                    repeat_patches: Vec::new(),
                    profile: StepProfile::default(),
                }));
            for &(actual, expected, words) in validators.get(&tile).into_iter().flatten() {
                programs[usize::from(tile)]
                    .steps
                    .push(TileStep::Compute(ComputeStep {
                        symbol: "ipu_stack_static_assert_equal_u32".into(),
                        output_address: TileAddress::Absolute(actual),
                        input_addresses: vec![
                            TileAddress::Absolute(actual),
                            TileAddress::Absolute(expected),
                        ],
                        arguments: vec![words],
                        profile: StepProfile::default(),
                    }));
            }
            programs[usize::from(tile)]
                .steps
                .push(TileStep::Compute(ComputeStep {
                    symbol: "ipu_stack_static_worker_delay".into(),
                    output_address: TileAddress::Absolute(DATA_BASE),
                    input_addresses: vec![TileAddress::Absolute(DATA_BASE)],
                    arguments: vec![rng.u32(1..=maximum_compute_delay)],
                    profile: StepProfile::default(),
                }));
        }
    }
    let data = buffers
        .into_iter()
        .map(|(tile, data)| TileProgramData {
            tile,
            address: DATA_BASE,
            data,
        })
        .collect::<Vec<_>>();
    let application = build_tile_program_package(&programs, &data, &[], toolchain, runtime_source)?;
    eprintln!(
        "exchangeStress seed={seed:#x} pattern={} cases={cases} transfers={} activeTiles={active_tiles} maxWords={maximum_words} maxTransfers={maximum_transfers} maxComputeDelay={maximum_compute_delay}",
        if overlap_sweep { "overlap" } else { "random" },
        transfers.len(),
    );
    Ok(StressPackage {
        application,
        active_tiles,
        transfers,
        rows: diagnostic_rows,
        readbacks: Vec::new(),
    })
}

pub(crate) fn build_phase_replay(
    compiled: &CompiledPackage,
    phase_index: usize,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<PhaseReplayPackage> {
    let phase = compiled
        .exchange_phases
        .get(phase_index)
        .with_context(|| format!("exchange phase {phase_index} is out of range"))?;
    build_physical_phase_replay(phase, phase_index, toolchain, runtime_source)
}

pub(crate) fn build_schedule_phase_replay(
    snapshot: &ipu_codegen::ExchangeScheduleSnapshot,
    phase_index: usize,
    first_transfer: usize,
    transfer_limit: Option<usize>,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<PhaseReplayPackage> {
    let mut problem = snapshot
        .phases
        .get(phase_index)
        .with_context(|| format!("exchange phase {phase_index} is out of range"))?
        .clone();
    if first_transfer > problem.transfers.len() {
        bail!("--exchange-replay-first-transfer is beyond the selected phase");
    }
    problem.transfers.drain(..first_transfer);
    if let Some(limit) = transfer_limit {
        if limit == 0 {
            bail!("--exchange-replay-transfer-limit must be nonzero");
        }
        problem.transfers.truncate(limit);
    }
    let scheduled = ipu_codegen::schedule_exchange_problem(snapshot.tile_count, &problem)?;
    build_physical_phase_replay(&scheduled.phase, phase_index, toolchain, runtime_source)
}

fn build_physical_phase_replay(
    phase: &ipu_codegen::PhysicalExchangePhase,
    phase_index: usize,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<PhaseReplayPackage> {
    let topology = Topology::c600();
    let execution_tiles = u16::try_from(topology.tile_count())?;
    let scheduled_tiles = u16::try_from(phase.programs.len())?;
    if phase.active.len() != usize::from(scheduled_tiles)
        || phase.incoming_bases.len() != usize::from(scheduled_tiles)
        || phase.activities.len() != usize::from(scheduled_tiles)
    {
        bail!("exchange phase {phase_index} has inconsistent per-tile metadata");
    }
    if phase
        .repeat_patches
        .iter()
        .any(|patches| !patches.is_empty())
    {
        bail!(
            "exchange phase {phase_index} uses repeat patches; replay a concrete iteration instead"
        );
    }

    let programs = (0..execution_tiles)
        .map(|tile| {
            let scheduled = tile < scheduled_tiles;
            let words = if scheduled {
                phase.programs[usize::from(tile)].clone()
            } else {
                inactive_exchange_program()
            };
            let row_address = ROW_BASE;
            Ok(TileProgram {
                tile,
                steps: vec![
                    TileStep::Exchange(ExchangeStep {
                        active: scheduled && phase.active[usize::from(tile)],
                        incoming_base: scheduled
                            .then(|| phase.incoming_bases[usize::from(tile)])
                            .unwrap_or(0),
                        preserve_base_registers: false,
                        incoming_mux: None,
                        incoming_format: 0,
                        incoming_mux_pair: None,
                        incoming_dcount: None,
                        sync_in_program: false,
                        program: PlacedExchangeRow {
                            address: row_address,
                            words,
                        },
                        setup_patch: None,
                        repeat_patches: Vec::new(),
                        profile: StepProfile::default(),
                    }),
                    // A patched breakpoint immediately following the final
                    // internal-exchange dispatch is not durable on IPU21.
                    // Cross one ordinary worker-call boundary before trapping;
                    // this occurs after the exchange epoch under test.
                    TileStep::Compute(ComputeStep {
                        symbol: "ipu_stack_static_worker_delay".into(),
                        output_address: TileAddress::Absolute(DATA_BASE),
                        input_addresses: vec![TileAddress::Absolute(DATA_BASE)],
                        arguments: vec![1],
                        profile: StepProfile::default(),
                    }),
                    TileStep::Checkpoint(CheckpointStep {
                        operation: u32::try_from(phase_index)?,
                        breakpoint: 0,
                        profile: StepProfile::default(),
                    }),
                ],
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut transfers = BTreeMap::<
        u32,
        (
            Option<(u16, ExchangeActivity)>,
            Vec<(u16, ExchangeActivity)>,
        ),
    >::new();
    let mut initial = vec![BTreeMap::<u32, u32>::new(); usize::from(scheduled_tiles)];
    for (tile, activities) in phase.activities.iter().enumerate() {
        let tile = u16::try_from(tile)?;
        for &activity in activities {
            if activity.address & 0b11 != 0 {
                bail!(
                    "phase {phase_index} transfer {} has unaligned address 0x{:x}",
                    activity.transfer,
                    activity.address
                );
            }
            let transfer = transfers.entry(activity.transfer).or_default();
            match activity.kind {
                ExchangeActivityKind::Send => {
                    if transfer.0.replace((tile, activity)).is_some() {
                        bail!(
                            "phase {phase_index} transfer {} has multiple senders",
                            activity.transfer
                        );
                    }
                }
                ExchangeActivityKind::Receive => transfer.1.push((tile, activity)),
                ExchangeActivityKind::PartnerBusy => {}
            }
            for word in 0..activity.words {
                let address = activity
                    .address
                    .checked_add(word.checked_mul(4).context("activity offset overflow")?)
                    .context("activity address overflow")?;
                initial[usize::from(tile)]
                    .entry(address)
                    .or_insert_with(|| replay_word(tile, address));
            }
        }
    }

    let transfers = transfers
        .into_iter()
        .map(|(id, (send, receives))| {
            let (source, send) =
                send.with_context(|| format!("phase {phase_index} transfer {id} has no sender"))?;
            if receives.is_empty() {
                bail!("phase {phase_index} transfer {id} has no receivers");
            }
            if receives
                .iter()
                .any(|(_, receive)| receive.words != send.words)
            {
                bail!("phase {phase_index} transfer {id} has inconsistent word counts");
            }
            Ok((id, source, ReplayTransfer { send, receives }))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut events = Vec::new();
    for &(transfer, source, ref replay) in &transfers {
        events.push((
            replay.send.start_cycle,
            1u8,
            ReplayEvent::Send {
                transfer,
                tile: source,
                address: replay.send.address,
                words: replay.send.words,
            },
        ));
        for &(tile, receive) in &replay.receives {
            events.push((
                receive.memory_end_cycle,
                0u8,
                ReplayEvent::Receive {
                    transfer,
                    tile,
                    address: receive.address,
                },
            ));
        }
    }
    events.sort_unstable_by_key(|&(cycle, order, _)| (cycle, order));

    let mut expected_memory = initial.clone();
    let mut payloads = BTreeMap::<u32, Vec<u32>>::new();
    for (_, _, event) in events {
        match event {
            ReplayEvent::Send {
                transfer,
                tile,
                address,
                words,
            } => {
                let memory = &expected_memory[usize::from(tile)];
                let payload = (0..words)
                    .map(|word| {
                        let address = address + word * 4;
                        memory.get(&address).copied().with_context(|| {
                            format!(
                                "phase {phase_index} transfer {transfer} reads untracked tile {tile} address 0x{address:x}"
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                payloads.insert(transfer, payload);
            }
            ReplayEvent::Receive {
                transfer,
                tile,
                address,
            } => {
                let payload = payloads.get(&transfer).with_context(|| {
                    format!(
                        "phase {phase_index} transfer {transfer} completes a receive before its send snapshot"
                    )
                })?;
                let memory = &mut expected_memory[usize::from(tile)];
                for (word, &value) in payload.iter().enumerate() {
                    memory.insert(address + u32::try_from(word)? * 4, value);
                }
            }
        }
    }

    let data = memory_spans(&initial)
        .into_iter()
        .map(|span| TileProgramData {
            tile: span.tile,
            address: span.address,
            data: span.words.into_iter().flat_map(u32::to_le_bytes).collect(),
        })
        .collect::<Vec<_>>();
    let expected = memory_spans(&expected_memory);
    let mut initial_origins = BTreeMap::<u32, Vec<(u16, u32)>>::new();
    for (tile, memory) in initial.iter().enumerate() {
        for (&address, &word) in memory {
            initial_origins
                .entry(word)
                .or_default()
                .push((u16::try_from(tile)?, address));
        }
    }
    let application = build_tile_program_package(&programs, &data, &[], toolchain, runtime_source)?;
    let overlapping_tiles = phase
        .activities
        .iter()
        .filter(|activities| {
            activities.iter().any(|send| {
                send.kind == ExchangeActivityKind::Send
                    && activities.iter().any(|receive| {
                        receive.kind == ExchangeActivityKind::Receive
                            && send.start_cycle < receive.end_cycle
                            && receive.start_cycle < send.end_cycle
                    })
            })
        })
        .count();
    eprintln!(
        "exchangeReplay phase={phase_index} transfers={} activeTiles={} overlappingTiles={} eventCycles={} touchedSpans={}",
        transfers.len(),
        phase.active.iter().filter(|&&active| active).count(),
        overlapping_tiles,
        phase.event_cycles,
        expected.len(),
    );
    Ok(PhaseReplayPackage {
        application,
        phase: phase_index,
        expected,
        activities: phase.activities.clone(),
        initial_origins,
    })
}

impl PhaseReplayPackage {
    pub(crate) fn service_readback(
        &self,
        device: &Device,
        sample_limit: usize,
        serviced: &mut bool,
    ) -> Result<()> {
        if *serviced {
            return Ok(());
        }
        let topology = Topology::c600();
        for tile in &self.application.tiles {
            if device.tile_context_state(u16::try_from(tile.physical_tile)?, 0)? != 2 {
                return Ok(());
            }
        }
        for tile in &self.application.tiles {
            let physical = u16::try_from(tile.physical_tile)?;
            let status = device.read_tile_context_status(physical, 0)?;
            let exception = TileException::from_status(status);
            if exception != TileException::PatchedBreak0 {
                bail!(
                    "exchange replay phase {} tile {physical} stopped with {exception} (status {status:#x})",
                    self.phase,
                );
            }
        }
        let samples = replay_samples(&self.expected, sample_limit);
        let mut checked = 0usize;
        for (tile, samples) in samples {
            let physical = topology.physical(tile)?;
            let addresses = samples
                .iter()
                .map(|&(address, _)| address)
                .collect::<Vec<_>>();
            let actual = device
                .read_tile_words_at_addresses_from_inactive_context(physical, 1, &addresses)
                .with_context(|| {
                    format!(
                        "read replay phase {} logical tile {} physical tile {} at {} sampled addresses",
                        self.phase, tile, physical, addresses.len(),
                    )
                })?;
            let differences = actual
                .iter()
                .zip(&samples)
                .filter(|(actual, (_, expected))| **actual != *expected)
                .take(16)
                .map(|(&actual, &(address, expected))| {
                    let roles = self.activities[usize::from(tile)]
                        .iter()
                        .filter(|activity| {
                            let end = activity.address.saturating_add(activity.words * 4);
                            (activity.address..end).contains(&address)
                        })
                        .map(|activity| {
                            (
                                activity.transfer,
                                activity.kind,
                                activity.start_cycle,
                                activity.end_cycle,
                                activity.memory_end_cycle,
                            )
                        })
                        .collect::<Vec<_>>();
                    let window = roles.iter().fold(None, |window, &(_, _, start, end, _)| {
                        Some(window.map_or((start, end), |(first, last): (u32, u32)| {
                            (first.min(start), last.max(end))
                        }))
                    });
                    let concurrent = window.map_or_else(Vec::new, |(start, end)| {
                        self.activities[usize::from(tile)]
                            .iter()
                            .filter(|activity| {
                                activity.start_cycle < end && start < activity.end_cycle
                            })
                            .map(|activity| {
                                (
                                    activity.transfer,
                                    activity.kind,
                                    activity.start_cycle,
                                    activity.end_cycle,
                                    activity.address,
                                    activity.words,
                                )
                            })
                            .collect::<Vec<_>>()
                    });
                    let transfer_ids = roles
                        .iter()
                        .map(|&(transfer, ..)| transfer)
                        .collect::<std::collections::BTreeSet<_>>();
                    let transfer_ids_ref = &transfer_ids;
                    let endpoints = self
                        .activities
                        .iter()
                        .enumerate()
                        .flat_map(|(endpoint_tile, activities)| {
                            activities.iter().filter_map(move |activity| {
                                transfer_ids_ref.contains(&activity.transfer).then_some((
                                    u16::try_from(endpoint_tile).unwrap(),
                                    activity.transfer,
                                    activity.kind,
                                    activity.start_cycle,
                                    activity.end_cycle,
                                    activity.address,
                                    activity.words,
                                ))
                            })
                        })
                        .collect::<Vec<_>>();
                    let endpoint_concurrent = endpoints
                        .iter()
                        .flat_map(|&(endpoint_tile, transfer, kind, start, end, _, _)| {
                            self.activities[usize::from(endpoint_tile)]
                                .iter()
                                .filter(move |activity| {
                                    activity.transfer != transfer
                                        && activity.start_cycle < end
                                        && start < activity.end_cycle
                                })
                                .map(move |activity| {
                                    (
                                        endpoint_tile,
                                        kind,
                                        transfer,
                                        activity.transfer,
                                        activity.kind,
                                        activity.start_cycle,
                                        activity.end_cycle,
                                        activity.address,
                                        activity.words,
                                    )
                                })
                        })
                        .collect::<Vec<_>>();
                    (
                        address,
                        expected,
                        actual,
                        self.initial_origins.get(&expected),
                        self.initial_origins.get(&actual),
                        roles,
                        concurrent,
                        endpoints,
                        endpoint_concurrent,
                    )
                })
                .collect::<Vec<_>>();
            if !differences.is_empty() {
                bail!(
                    "exchange replay phase {} corrupted logical tile {tile}: {differences:?}",
                    self.phase,
                );
            }
            checked += samples.len();
        }
        eprintln!("exchangeReplay phase={} sampledWords={checked}", self.phase);
        const IPU21_NOP_INSTRUCTION: u32 = 0x19e0_0000;
        for tile in &self.application.tiles {
            let physical = u16::try_from(tile.physical_tile)?;
            let pc = device.read_tile_program_counter(physical, 0)?;
            device.write_tile_word_from_stopped_context(physical, 0, pc, IPU21_NOP_INSTRUCTION)?;
            device.clear_tile_exception(physical, 0)?;
        }
        *serviced = true;
        Ok(())
    }
}

fn replay_samples(expected: &[ExpectedSpan], limit: usize) -> BTreeMap<u16, Vec<(u32, u32)>> {
    let total_words = expected.iter().map(|span| span.words.len()).sum::<usize>();
    let wanted = limit.min(total_words);
    let mut selected = std::collections::BTreeSet::<(usize, usize)>::new();
    if wanted == 0 || expected.is_empty() {
        return BTreeMap::new();
    }
    let first_span_samples = wanted.min(expected.len());
    for sample in 0..first_span_samples {
        let span = sample * expected.len() / first_span_samples;
        selected.insert((span, 0));
    }
    if selected.len() < wanted && first_span_samples == expected.len() {
        for (span, values) in expected.iter().enumerate() {
            if selected.len() == wanted {
                break;
            }
            selected.insert((span, values.words.len() - 1));
        }
    }
    let mut ends = Vec::with_capacity(expected.len());
    let mut end = 0usize;
    for span in expected {
        end += span.words.len();
        ends.push(end);
    }
    let attempts = wanted.saturating_mul(4).max(wanted);
    for sample in 0..attempts {
        if selected.len() == wanted {
            break;
        }
        let linear = sample * total_words / attempts;
        let span = ends.partition_point(|&end| end <= linear);
        let start = span.checked_sub(1).map_or(0, |previous| ends[previous]);
        selected.insert((span, linear - start));
    }
    let mut result = BTreeMap::<u16, Vec<(u32, u32)>>::new();
    for (span, word) in selected {
        let span = &expected[span];
        result.entry(span.tile).or_default().push((
            span.address + u32::try_from(word).expect("tile word index fits u32") * 4,
            span.words[word],
        ));
    }
    for samples in result.values_mut() {
        samples.sort_unstable_by_key(|&(address, _)| address);
    }
    result
}

fn memory_spans(memory: &[BTreeMap<u32, u32>]) -> Vec<ExpectedSpan> {
    let mut spans = Vec::new();
    for (tile, words) in memory.iter().enumerate() {
        let mut current: Option<ExpectedSpan> = None;
        for (&address, &word) in words {
            let contiguous = current.as_ref().is_some_and(|span| {
                span.address + u32::try_from(span.words.len()).unwrap() * 4 == address
            });
            if contiguous {
                current.as_mut().unwrap().words.push(word);
            } else {
                if let Some(span) = current.take() {
                    spans.push(span);
                }
                current = Some(ExpectedSpan {
                    tile: u16::try_from(tile).expect("tile count was supplied as u16"),
                    address,
                    words: vec![word],
                });
            }
        }
        if let Some(span) = current {
            spans.push(span);
        }
    }
    spans
}

fn replay_word(tile: u16, address: u32) -> u32 {
    0xa5a5_5a5a ^ u32::from(tile).wrapping_mul(0x9e37_79b9) ^ (address >> 2).rotate_left(13)
}

impl StressPackage {
    pub(crate) fn live_exchange_state(&self, runtime: &Runtime) -> String {
        let topology = Topology::c600();
        let mut states = Vec::new();
        let mut relevant = Vec::new();
        for transfer in &self.transfers {
            relevant.push(transfer.source);
            relevant.extend(transfer.destinations.iter().copied());
            if let Ok(paired) = topology.paired_logical(transfer.source) {
                relevant.push(paired);
            }
        }
        relevant.sort_unstable();
        relevant.dedup();
        for &logical in &relevant {
            if logical >= self.active_tiles {
                continue;
            }
            let Ok(physical) = topology.physical(logical) else {
                continue;
            };
            let context = runtime.device().tile_context_state(physical, 0);
            let error = runtime.device().tile_exchange_receive_error(physical);
            let exchange = runtime.device().tile_exchange_state(physical);
            let stopped = context.as_ref().is_ok_and(|state| matches!(*state, 2 | 3));
            let status = stopped.then(|| runtime.device().read_tile_context_status(physical, 0));
            let pc = stopped.then(|| runtime.device().read_tile_program_counter(physical, 0));
            states.push(format!(
                "logical={logical} physical={physical} context={context:?} ererr={error:?} exchange={exchange:?} status={status:?} pc={pc:?}"
            ));
        }
        states.join("; ")
    }

    pub(crate) fn validate_readbacks(&self, output: &[u8]) -> Result<()> {
        let mut offset = 0usize;
        for span in &self.readbacks {
            let bytes = span.words.len() * 4;
            let actual = output
                .get(offset..offset + bytes)
                .context("paired exchange host output is truncated")?
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte word")))
                .collect::<Vec<_>>();
            offset += bytes;
            let differences = span
                .words
                .iter()
                .zip(&actual)
                .enumerate()
                .filter(|(_, (expected, actual))| expected != actual)
                .take(16)
                .map(|(word, (&expected, &actual))| (word, expected, actual))
                .collect::<Vec<_>>();
            if !differences.is_empty() {
                bail!(
                    "paired exchange corrupted logical tile {} at 0x{:x}: {differences:?}",
                    span.tile,
                    span.address,
                );
            }
        }
        if !self.readbacks.is_empty() {
            eprintln!(
                "exchangeWide hardwareReadback=PASS spans={} words={}",
                self.readbacks.len(),
                self.readbacks
                    .iter()
                    .map(|span| span.words.len())
                    .sum::<usize>(),
            );
        }
        Ok(())
    }

    pub(crate) fn static_diagnostic(&self, case: u32) -> Result<String> {
        let row = self
            .rows
            .iter()
            .find(|row| row.case == Some(case))
            .with_context(|| format!("exchange diagnostic case {case} is out of range"))?;
        let transfers = self
            .transfers
            .iter()
            .filter(|transfer| transfer.case == case)
            .map(|transfer| {
                format!(
                    "source={} address=0x{:x} destinations={:?} addresses={:?} words={} requestedOffset={} encodedOffset={} sender={}..{} receivers={:?}..{:?}",
                    transfer.source,
                    transfer.source_address,
                    transfer.destinations,
                    transfer.destination_addresses,
                    transfer.words,
                    transfer.requested_schedule_offset,
                    transfer.schedule_offset,
                    transfer.timing.payload_start,
                    transfer.timing.payload_end,
                    transfer.timing.receiver_payload_starts,
                    transfer.timing.receiver_payload_ends,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let programs = row
            .programs
            .iter()
            .map(|(&tile, program)| {
                let decoded =
                    ipu_exchange::diagnostic::diagnose_plan_program(program, Some(row.address))?;
                Ok(format!(
                    "tile={tile} words={} events={}\n{}",
                    program.len(),
                    decoded.event_cycles,
                    decoded.render()
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .join("");
        Ok(format!(
            "exchangeCase={case} row=0x{:x}..0x{:x}\ntransfers:\n{transfers}\nprograms:\n{programs}",
            row.address, row.end
        ))
    }

    pub(crate) fn failure_context(&self, runtime: &Runtime) -> String {
        let topology = Topology::c600();
        let mut stopped = Vec::new();
        for logical in 0..self.active_tiles {
            let Ok(physical) = topology.physical(logical) else {
                continue;
            };
            let Ok(pc) = runtime.device().read_tile_program_counter(physical, 0) else {
                continue;
            };
            if let Some(row) = self
                .rows
                .iter()
                .find(|row| (row.address..row.end).contains(&pc))
            {
                stopped.push((logical, physical, row.case, pc));
            }
        }
        let cases = stopped
            .iter()
            .filter_map(|entry| entry.2)
            .collect::<Vec<_>>();
        let transfers = self
            .transfers
            .iter()
            .filter(|transfer| cases.contains(&transfer.case))
            .map(|transfer| {
                (
                    transfer.case,
                    transfer.source,
                    &transfer.destinations,
                    transfer.words,
                    transfer.source_address,
                    &transfer.destination_addresses,
                )
            })
            .collect::<Vec<_>>();
        let rows = stopped
            .iter()
            .filter_map(|&(logical, physical, case, pc)| {
                let row = self.rows.iter().find(|row| row.case == case)?;
                let expected = row.programs.get(&logical)?;
                let actual = runtime
                    .device()
                    .read_tile_words_from_inactive_context(
                        physical,
                        1,
                        row.address,
                        u32::try_from(expected.len()).ok()?,
                    )
                    .ok()?;
                let differences = expected
                    .iter()
                    .zip(&actual)
                    .enumerate()
                    .filter(|(_, (expected, actual))| expected != actual)
                    .map(|(offset, (&expected, &actual))| (offset, expected, actual))
                    .collect::<Vec<_>>();
                let decode =
                    ipu_exchange::diagnostic::diagnose_plan_program(expected, Some(row.address))
                        .map(|diagnostic| diagnostic.render_around_address(pc, 16));
                Some((
                    logical,
                    physical,
                    case,
                    pc,
                    differences.len(),
                    differences.into_iter().take(16).collect::<Vec<_>>(),
                    decode,
                ))
            })
            .collect::<Vec<_>>();
        format!(
            "random exchange failure; stoppedRows={stopped:?}; transfers={transfers:?}; rowDiagnostics={rows:?}"
        )
    }
}

fn allocate(
    cursors: &mut [u32],
    tile: u16,
    bytes: u32,
    limit: u32,
    rng: &mut fastrand::Rng,
) -> Result<u32> {
    let cursor = &mut cursors[usize::from(tile)];
    *cursor = cursor
        .checked_add(rng.u32(0..=8) * 4)
        .context("test allocation overflow")?;
    let address = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .context("test allocation overflow")?;
    if *cursor > limit {
        bail!(
            "random stress data exhausted tile {tile}; reduce --exchange-cases or --exchange-max-words"
        );
    }
    Ok(address)
}

fn allocate_with_fixed_padding(
    cursors: &mut [u32],
    tile: u16,
    bytes: u32,
    padding: u32,
    limit: u32,
) -> Result<u32> {
    let cursor = &mut cursors[usize::from(tile)];
    let address = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .and_then(|cursor| cursor.checked_add(padding))
        .context("test allocation overflow")?;
    if *cursor > limit {
        bail!(
            "random stress data exhausted tile {tile}; reduce --exchange-cases or --exchange-max-words"
        );
    }
    Ok(address)
}

fn random_words(rng: &mut fastrand::Rng, maximum: u32) -> u32 {
    const EDGES: &[u32] = &[
        1, 2, 3, 15, 16, 31, 32, 51, 52, 53, 63, 64, 65, 127, 128, 255, 256, 511, 512, 1023, 1024,
        4095, 4148,
    ];
    if rng.bool() {
        let eligible = EDGES.partition_point(|&words| words <= maximum);
        EDGES[rng.usize(0..eligible)]
    } else {
        rng.u32(1..=maximum)
    }
}

fn overlap_specs(
    topology: &Topology,
    case: u32,
    tiles: &[u16],
    maximum_words: u32,
    rng: &mut fastrand::Rng,
) -> Result<Vec<TransferSpec>> {
    let [incoming_source, pivot, outgoing_destination] = *tiles else {
        bail!("overlap case requires exactly three tiles");
    };
    let words = random_words(rng, maximum_words);
    let incoming = point_plan(topology, incoming_source, pivot, words)?;
    let outgoing = point_plan(topology, pivot, outgoing_destination, words)?;
    let empty = PhaseProgramBuilder::new(u16::try_from(topology.tile_count())?);
    let incoming_base = empty.transfer_timing_at(incoming_source, &[pivot], &incoming, 0, words)?;
    let outgoing_base =
        empty.transfer_timing_at(pivot, &[outgoing_destination], &outgoing, 0, words)?;
    let incoming_start = incoming_base.receiver_payload_starts[0];
    let outgoing_start = outgoing_base.payload_start;
    let anchor = incoming_start.max(outgoing_start);
    let maximum_delta = words.saturating_sub(1);
    let deltas = [
        0,
        1.min(maximum_delta),
        maximum_delta / 4,
        maximum_delta / 2,
        maximum_delta.saturating_sub(1),
        maximum_delta,
    ];
    let delta = deltas[usize::try_from(case)? % deltas.len()];
    let incoming_first = case & 1 == 0;
    let incoming_target = anchor + if incoming_first { 0 } else { delta };
    let outgoing_target = anchor + if incoming_first { delta } else { 0 };
    let incoming = TransferSpec {
        source: incoming_source,
        destinations: vec![pivot],
        words,
        schedule_offset: Some(incoming_target - incoming_start),
    };
    let outgoing = TransferSpec {
        source: pivot,
        destinations: vec![outgoing_destination],
        words,
        schedule_offset: Some(outgoing_target - outgoing_start),
    };
    Ok(if incoming_first {
        vec![incoming, outgoing]
    } else {
        vec![outgoing, incoming]
    })
}

fn point_plan(
    topology: &Topology,
    source: u16,
    destination: u16,
    words: u32,
) -> Result<MulticastPlan> {
    let point = topology.point_to_point(source, destination, words)?;
    Ok(MulticastPlan {
        sender: point.sender,
        receivers: vec![finalize_point_receiver(
            &point.receivers[0],
            topology.physical(source)?,
        )?],
    })
}

fn paired_control_words(
    topology: &Topology,
    source: u16,
    receiver: u16,
    maximum: u32,
) -> Result<Option<u32>> {
    let plan = topology.point_to_point(source, receiver, 1)?;
    let receiver = finalize_point_receiver(&plan.receivers[0], topology.physical(source)?)?;
    let timing = scheduled_receiver_timing(&receiver, 0)?;
    Ok(timing
        .pointer_event
        .and_then(|pointer| pointer.checked_sub(timing.source_event))
        .filter(|words| (1..=maximum).contains(words)))
}

fn word_value(case: u32, source: u16, destinations: &[u16], index: u32) -> u32 {
    let mut value =
        0x9e37_79b9u32 ^ case.wrapping_mul(0x85eb_ca6b) ^ index.wrapping_mul(0xc2b2_ae35);
    value ^= u32::from(source) << 16;
    for &destination in destinations {
        value = value.rotate_left(5) ^ u32::from(destination);
    }
    value
}

fn fill_source(buffer: &mut [u8], address: u32, payload: &Payload) {
    let offset = (address - DATA_BASE) as usize;
    for index in 0..payload.words {
        let start = offset + index as usize * 4;
        buffer[start..start + 4].copy_from_slice(
            &word_value(payload.case, payload.source, &payload.destinations, index).to_le_bytes(),
        );
    }
}
