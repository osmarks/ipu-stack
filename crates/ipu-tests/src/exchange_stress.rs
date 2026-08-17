use anyhow::{Context, Result, bail};
use ipu_codegen::{TileProgramData, build_tile_program_package, inactive_exchange_program};
use ipu_elf::Toolchain;
use ipu_package::{Application, Binding, RegionSlice};
use ipu_runtime::Runtime;
use ipu_target::exchange::{
    PhaseProgramBuilder, PhaseTransferTiming, PhysicalTransfer, TransferEndpoint, TransferWidth,
    finalize_point_receiver, scheduled_receiver_timing,
};
use ipu_target::instruction::encode_exchange_delay;
use ipu_target::program::{
    ComputeStep, ExchangeStep, PlacedExchangeRow, StepProfile, TileAddress, TileProgram, TileStep,
};
use ipu_target::topology::Topology;
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
struct StressTransfer {
    case: u32,
    physical: PhysicalTransfer,
    expected_words: Vec<u32>,
    requested_schedule_offset: u32,
    schedule_offset: u32,
    timing: PhaseTransferTiming,
}

pub(crate) struct StressPackage {
    pub application: Application,
    active_tiles: u16,
    transfers: Vec<StressTransfer>,
    rows: Vec<StressRow>,
    readbacks: Vec<ExpectedSpan>,
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
    let topology = ipu_target::hardware::HardwareTarget::Ipu21.topology();
    let execution_tiles = u16::try_from(topology.tile_count())?;
    if words < 128
        || words & 1 != 0
        || words / 2
            > ipu_target::hardware::HardwareTarget::Ipu21
                .exchange()
                .maximum_transfer_words
    {
        bail!("paired 64-bit exchange payload must contain 128..=8296 even u32 words");
    }
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
        ipu_target::instruction::SYNC_SUPERVISOR_INSTRUCTION,
        encode_exchange_delay(0),
        ipu_target::instruction::RETURN_M10_INSTRUCTION,
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
        let source_element_size =
            if source_base >= ipu_target::memory::IPU21_INTERLEAVED_MEMORY_BASE {
                ipu_target::memory::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_target::memory::TILE_MEMORY_ELEMENT_SIZE
            };
        let destination_element_size =
            if destination_base >= ipu_target::memory::IPU21_INTERLEAVED_MEMORY_BASE {
                ipu_target::memory::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_target::memory::TILE_MEMORY_ELEMENT_SIZE
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

        let transfer = PhysicalTransfer {
            source,
            source_addresses: vec![source_address],
            destinations: destinations
                .iter()
                .map(|&tile| TransferEndpoint(tile, destination_address))
                .collect(),
            words,
            width: TransferWidth::Paired64,
        };
        let transfer = transfer.resolve(&topology, None)?;
        let mut builder = PhaseProgramBuilder::new(execution_tiles);
        let schedule_offset = builder.earliest_transfer_offset(&transfer, 0)?;
        let timing = builder.append_transfer_at(&transfer, schedule_offset)?;
        let phase = builder.finish()?;
        let mut rows = phase.programs;
        for row in rows.iter_mut().flatten() {
            row.insert(0, ipu_target::instruction::SYNC_SUPERVISOR_INSTRUCTION);
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
                        ipu_target::instruction::SYNC_SUPERVISOR_INSTRUCTION,
                        encode_exchange_delay(0),
                        ipu_target::instruction::RETURN_M10_INSTRUCTION,
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
        transfers.push(StressTransfer {
            case: u32::try_from(case)?,
            physical: PhysicalTransfer {
                source,
                source_addresses: vec![source_address],
                destinations: destinations
                    .iter()
                    .map(|&tile| TransferEndpoint(tile, destination_address))
                    .collect(),
                words,
                width: TransferWidth::Paired64,
            },
            expected_words: payload.clone(),
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
    let maximum_tiles = ipu_target::hardware::HardwareTarget::Ipu21
        .topology()
        .tile_count();
    if active_tiles < 2 || usize::from(active_tiles) > maximum_tiles {
        bail!("exchange stress requires 2..={maximum_tiles} active tiles");
    }
    if overlap_sweep && active_tiles < 3 {
        bail!("the exchange overlap sweep requires at least three active tiles");
    }
    if cases == 0 {
        bail!("--exchange-cases must be nonzero");
    }
    if maximum_words == 0
        || maximum_words
            > ipu_target::hardware::HardwareTarget::Ipu21
                .exchange()
                .maximum_transfer_words
    {
        bail!(
            "--exchange-max-words must be in 1..={}",
            ipu_target::hardware::HardwareTarget::Ipu21
                .exchange()
                .maximum_transfer_words
        );
    }
    if maximum_transfers == 0 {
        bail!("--exchange-max-transfers must be nonzero");
    }
    if maximum_compute_delay == 0 {
        bail!("--exchange-compute-delay must be nonzero");
    }
    let topology = ipu_target::hardware::HardwareTarget::Ipu21.topology();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut destination_cursors = vec![DATA_BASE; usize::from(active_tiles)];
    let mut source_cursors = vec![SOURCE_BASE; usize::from(active_tiles)];
    let mut expected_cursors = vec![EXPECTED_BASE; usize::from(active_tiles)];
    let mut buffers = BTreeMap::<u16, Vec<u8>>::new();
    let mut transfers = Vec::new();
    let mut available_payloads = vec![Vec::<(u32, Vec<u32>)>::new(); usize::from(active_tiles)];
    let mut phase_rows = Vec::with_capacity(cases as usize);
    let mut diagnostic_rows = Vec::with_capacity(cases as usize);
    let mut row_address = ROW_BASE;
    let mut previous_shape: Option<Vec<(PhysicalTransfer, Option<u32>)>> = None;

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
                        (
                            transfer_shape(sources[(index / 2) % sources.len()], [receiver], words),
                            None,
                        )
                    } else {
                        (
                            transfer_shape(
                                receiver,
                                [sources[(index / 2 + 1) % sources.len()]],
                                words,
                            ),
                            None,
                        )
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
                shape.push((
                    transfer_shape(source, destinations, random_words(&mut rng, maximum_words)),
                    None,
                ));
            }
            previous_shape = Some(shape.clone());
            shape
        } else {
            shape
        };
        let mut builder = PhaseProgramBuilder::new(u16::try_from(topology.tile_count())?);
        let mut validators = BTreeMap::<u16, Vec<(u32, u32, u32)>>::new();
        for (mut physical, schedule_offset) in shape {
            let source = physical.source;
            let destinations = physical
                .destinations
                .iter()
                .map(|endpoint| endpoint.0)
                .collect::<Vec<_>>();
            let mut words = physical.words;
            let chained = (contiguous_receiver.is_none()
                && !available_payloads[usize::from(source)].is_empty()
                && rng.usize(0..4) == 0)
                .then(|| {
                    let candidates = &available_payloads[usize::from(source)];
                    candidates[rng.usize(0..candidates.len())].clone()
                });
            let (source_address, payload) = if let Some((address, payload)) = chained {
                words = u32::try_from(payload.len())?;
                (address, payload)
            } else {
                let payload = (0..words)
                    .map(|index| word_value(case, source, &destinations, index))
                    .collect::<Vec<_>>();
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
            physical.source_addresses[0] = source_address;
            physical.words = words;
            for (endpoint, &address) in physical.destinations.iter_mut().zip(&destination_addresses)
            {
                endpoint.1 = address;
            }
            let incoming_base = (destinations.len() == 1).then_some(destination_addresses[0]);
            let transfer = physical.resolve(&topology, incoming_base)?;
            let requested_schedule_offset = schedule_offset.unwrap_or(0);
            let schedule_offset =
                builder.earliest_transfer_offset(&transfer, requested_schedule_offset)?;
            let timing = builder
                .append_transfer_at(&transfer, schedule_offset)
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
            transfers.push(StressTransfer {
                case,
                physical,
                expected_words: payload,
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

impl StressPackage {
    pub(crate) fn live_exchange_state(&self, runtime: &Runtime) -> String {
        let topology = ipu_target::hardware::HardwareTarget::Ipu21.topology();
        let mut states = Vec::new();
        let mut relevant = Vec::new();
        for transfer in &self.transfers {
            relevant.push(transfer.physical.source);
            relevant.extend(transfer.physical.destination_tiles());
            if let Ok(paired) = topology.paired_logical(transfer.physical.source) {
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
                    "source={} address=0x{:x} destinations={:?} addresses={:?} words={} expectedWords={} requestedOffset={} encodedOffset={} sender={}..{} receivers={:?}..{:?}",
                    transfer.physical.source,
                    transfer.physical.source_address(),
                    transfer.physical.destination_tiles().collect::<Vec<_>>(),
                    transfer.physical.destination_addresses().collect::<Vec<_>>(),
                    transfer.physical.words,
                    transfer.expected_words.len(),
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
                    ipu_target::exchange::parse::diagnose_plan_program(program, Some(row.address))?;
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
        let topology = ipu_target::hardware::HardwareTarget::Ipu21.topology();
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
                    transfer.physical.source,
                    transfer.physical.destination_tiles().collect::<Vec<_>>(),
                    transfer.physical.words,
                    transfer.physical.source_address(),
                    transfer
                        .physical
                        .destination_addresses()
                        .collect::<Vec<_>>(),
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
                    ipu_target::exchange::parse::diagnose_plan_program(expected, Some(row.address))
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
) -> Result<Vec<(PhysicalTransfer, Option<u32>)>> {
    let [incoming_source, pivot, outgoing_destination] = *tiles else {
        bail!("overlap case requires exactly three tiles");
    };
    let words = random_words(rng, maximum_words);
    let incoming = point_transfer(topology, incoming_source, pivot, words)?;
    let outgoing = point_transfer(topology, pivot, outgoing_destination, words)?;
    let empty = PhaseProgramBuilder::new(u16::try_from(topology.tile_count())?);
    let incoming_base = empty.transfer_timing_at(&incoming, 0)?;
    let outgoing_base = empty.transfer_timing_at(&outgoing, 0)?;
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
    let incoming = (
        transfer_shape(incoming_source, [pivot], words),
        Some(incoming_target - incoming_start),
    );
    let outgoing = (
        transfer_shape(pivot, [outgoing_destination], words),
        Some(outgoing_target - outgoing_start),
    );
    Ok(if incoming_first {
        vec![incoming, outgoing]
    } else {
        vec![outgoing, incoming]
    })
}

fn transfer_shape(
    source: u16,
    destinations: impl IntoIterator<Item = u16>,
    words: u32,
) -> PhysicalTransfer {
    PhysicalTransfer {
        source,
        source_addresses: vec![0],
        destinations: destinations
            .into_iter()
            .map(|tile| TransferEndpoint(tile, 0))
            .collect(),
        words,
        width: TransferWidth::Word32,
    }
}

fn point_transfer(
    topology: &Topology,
    source: u16,
    destination: u16,
    words: u32,
) -> Result<ipu_target::exchange::ResolvedTransfer> {
    Ok(PhysicalTransfer {
        source,
        source_addresses: vec![DATA_BASE],
        destinations: vec![TransferEndpoint(destination, DATA_BASE)],
        words,
        width: TransferWidth::Word32,
    }
    .resolve(topology, Some(DATA_BASE))?)
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

fn fill_source(buffer: &mut [u8], address: u32, payload: &[u32]) {
    let offset = (address - DATA_BASE) as usize;
    for (index, &word) in payload.iter().enumerate() {
        let start = offset + index * 4;
        buffer[start..start + 4].copy_from_slice(&word.to_le_bytes());
    }
}
