use anyhow::{Context, Result, bail};
use ipu_codegen::{
    ComputeStep, ExchangeStep, PlacedExchangeRow, StepProfile, TileAddress, TileProgram,
    TileProgramData, TileStep, build_tile_program_package, inactive_exchange_program,
};
use ipu_elf::Toolchain;
use ipu_exchange::{
    MulticastPlan, PhaseProgramBuilder, Topology, finalize_point_receiver, patch_receiver_address,
    patch_sender_address, scheduled_receiver_timing,
};
use ipu_package::Application;
use ipu_runtime::Runtime;
use std::collections::BTreeMap;
use std::path::Path;

const DATA_BASE: u32 = 0x60000;
const SOURCE_BASE: u32 = 0x65000;
const EXPECTED_BASE: u32 = 0x6c000;
const DATA_LIMIT: u32 = 0x73800;
const ROW_BASE: u32 = 0x5c000;

#[derive(Clone, Debug)]
struct Transfer {
    case: u32,
    source: u16,
    destinations: Vec<u16>,
    source_address: u32,
    destination_addresses: Vec<u32>,
    words: u32,
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
}

#[derive(Clone, Debug)]
struct StressRow {
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
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<StressPackage> {
    let maximum_tiles = Topology::c600().tile_count();
    if active_tiles < 2 || usize::from(active_tiles) > maximum_tiles {
        bail!("exchange stress requires 2..={maximum_tiles} active tiles");
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
    let mut previous_shape: Option<Vec<(u16, Vec<u16>, u32)>> = None;

    for case in 0..cases {
        let mut tiles = (0..active_tiles).collect::<Vec<_>>();
        rng.shuffle(&mut tiles);
        let group_tiles = rng.usize(2..=8).min(tiles.len());
        let group = &tiles[..group_tiles];
        let contiguous_receiver = (case == 0).then_some(group[0]);
        let shape = if let Some(receiver) = contiguous_receiver {
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
                        (sources[(index / 2) % sources.len()], vec![receiver], words)
                    } else {
                        (
                            receiver,
                            vec![sources[(index / 2 + 1) % sources.len()]],
                            words,
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
                shape.push((source, destinations, random_words(&mut rng, maximum_words)));
            }
            previous_shape = Some(shape.clone());
            shape
        } else {
            shape
        };
        let mut builder = PhaseProgramBuilder::new(u16::try_from(topology.tile_count())?);
        let mut validators = BTreeMap::<u16, Vec<(u32, u32, u32)>>::new();
        for (source, destinations, mut words) in shape {
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
                        &point.receiver,
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
            let schedule_offset =
                builder.earliest_transfer_offset(source, &destinations, &plan, words, 0)?;
            builder.append_transfer_at(source, &destinations, &plan, schedule_offset, words)?;
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
    let application = build_tile_program_package(&programs, &data, toolchain, runtime_source)?;
    eprintln!(
        "exchangeStress seed={seed:#x} cases={cases} transfers={} activeTiles={active_tiles} maxWords={maximum_words} maxTransfers={maximum_transfers} maxComputeDelay={maximum_compute_delay}",
        transfers.len(),
    );
    Ok(StressPackage {
        application,
        active_tiles,
        transfers,
        rows: diagnostic_rows,
    })
}

impl StressPackage {
    pub(crate) fn static_diagnostic(&self, case: u32) -> Result<String> {
        let row = self
            .rows
            .get(usize::try_from(case)?)
            .with_context(|| format!("exchange diagnostic case {case} is out of range"))?;
        let transfers = self
            .transfers
            .iter()
            .filter(|transfer| transfer.case == case)
            .map(|transfer| {
                format!(
                    "source={} address=0x{:x} destinations={:?} addresses={:?} words={}",
                    transfer.source,
                    transfer.source_address,
                    transfer.destinations,
                    transfer.destination_addresses,
                    transfer.words,
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
            if let Some((case, _)) = self
                .rows
                .iter()
                .enumerate()
                .find(|(_, row)| (row.address..row.end).contains(&pc))
            {
                stopped.push((logical, physical, case, pc));
            }
        }
        let cases = stopped
            .iter()
            .map(|entry| entry.2 as u32)
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
                let row = self.rows.get(case)?;
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

fn paired_control_words(
    topology: &Topology,
    source: u16,
    receiver: u16,
    maximum: u32,
) -> Result<Option<u32>> {
    let plan = topology.point_to_point(source, receiver, 1)?;
    let receiver = finalize_point_receiver(&plan.receiver, topology.physical(source)?)?;
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
