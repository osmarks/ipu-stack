use anyhow::{Context, Result, bail};
use ipu_codegen::{
    ComputeStep, ExchangeStep, PlacedExchangeRow, StepProfile, TileAddress, TileProgram,
    TileProgramData, TileStep, build_tile_program_package, inactive_exchange_program,
};
use ipu_elf::Toolchain;
use ipu_exchange::{
    MulticastPlan, PlanProgramBuilder, Topology, finalize_point_receiver, patch_receiver_address,
    patch_sender_address, plan_event_cycles,
};
use ipu_package::Application;
use ipu_runtime::Runtime;
use std::collections::BTreeMap;
use std::path::Path;

const ACTIVE_TILES: u16 = 64;
const DATA_BASE: u32 = 0x60000;
const DATA_LIMIT: u32 = 0x73800;
const ROW_BASE: u32 = 0x59000;

#[derive(Clone, Debug)]
struct Transfer {
    case: u32,
    source: u16,
    destinations: Vec<u16>,
    source_address: u32,
    destination_addresses: Vec<u32>,
    words: u32,
}

pub(crate) struct StressPackage {
    pub application: Application,
    transfers: Vec<Transfer>,
    row_ranges: Vec<(u32, u32)>,
}

pub(crate) fn build(
    seed: u64,
    cases: u32,
    maximum_words: u32,
    maximum_compute_delay: u32,
    toolchain: &Toolchain,
    runtime_source: &Path,
) -> Result<StressPackage> {
    if cases == 0 {
        bail!("--exchange-cases must be nonzero");
    }
    if maximum_words == 0 || maximum_words > ipu_exchange::MAX_TRANSFER_WORDS {
        bail!(
            "--exchange-max-words must be in 1..={}",
            ipu_exchange::MAX_TRANSFER_WORDS
        );
    }
    if maximum_compute_delay == 0 {
        bail!("--exchange-compute-delay must be nonzero");
    }
    let topology = Topology::c600();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut cursors = vec![DATA_BASE; usize::from(ACTIVE_TILES)];
    let mut buffers = vec![vec![0u8; (DATA_LIMIT - DATA_BASE) as usize]; usize::from(ACTIVE_TILES)];
    let mut transfers = Vec::new();
    let mut phase_rows = Vec::with_capacity(cases as usize);
    let mut row_ranges = Vec::with_capacity(cases as usize);
    let mut row_address = ROW_BASE;
    let mut previous_shape: Option<Vec<(u16, Vec<u16>, u32)>> = None;

    for case in 0..cases {
        let mut tiles = (0..ACTIVE_TILES).collect::<Vec<_>>();
        rng.shuffle(&mut tiles);
        let group_tiles = rng.usize(2..=8).min(tiles.len());
        let group = &tiles[..group_tiles];
        let shape = if rng.bool() {
            previous_shape.clone().unwrap_or_default()
        } else {
            Vec::new()
        };
        let shape = if shape.is_empty() {
            let mut shape = Vec::new();
            for _ in 0..rng.usize(1..=4) {
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
        let mut builders = BTreeMap::<u16, PlanProgramBuilder>::new();
        let mut validators = BTreeMap::<u16, Vec<(u32, u32, u32)>>::new();
        let mut horizon = 0u32;
        for (source, destinations, words) in shape {
            let schedule_offset = if horizon == 0 { 0 } else { horizon + 1 };
            let bytes = words * 4;
            let source_address = allocate(&mut cursors, source, bytes, &mut rng)?;
            let destination_addresses = destinations
                .iter()
                .map(|&tile| allocate(&mut cursors, tile, bytes, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let expected_addresses = destinations
                .iter()
                .map(|&tile| allocate(&mut cursors, tile, bytes, &mut rng))
                .collect::<Result<Vec<_>>>()?;
            let mut plan = if destinations.len() == 1 && schedule_offset == 0 {
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
            builders
                .entry(source)
                .or_default()
                .append_scheduled_row_at(&plan.sender, schedule_offset)?;
            for (&tile, row) in destinations.iter().zip(&plan.receivers) {
                builders
                    .entry(tile)
                    .or_default()
                    .append_scheduled_row_at(row, schedule_offset)?;
            }
            horizon = horizon.max(plan_event_cycles(&plan.sender)? + schedule_offset);
            for row in &plan.receivers {
                horizon = horizon.max(plan_event_cycles(row)? + schedule_offset);
            }
            fill_source(
                &mut buffers[usize::from(source)],
                source_address,
                case,
                source,
                &destinations,
                words,
            );
            for ((&tile, &destination), &expected) in destinations
                .iter()
                .zip(&destination_addresses)
                .zip(&expected_addresses)
            {
                fill_source(
                    &mut buffers[usize::from(tile)],
                    expected,
                    case,
                    source,
                    &destinations,
                    words,
                );
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
        let rows = (0..topology.tile_count() as u16)
            .map(|tile| {
                builders
                    .remove(&tile)
                    .map(|builder| builder.finish())
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        row_ranges.push((row_address, row_end));
        phase_rows.push((row_address, rows, validators, horizon));
        row_address = (row_end + 3) & !3;
    }

    let execution_tiles = u16::try_from(topology.tile_count())?;
    let mut programs = (0..execution_tiles)
        .map(|tile| TileProgram {
            tile,
            steps: Vec::with_capacity(cases as usize),
        })
        .collect::<Vec<_>>();
    for (address, rows, validators, horizon) in phase_rows {
        for tile in 0..execution_tiles {
            let row = rows[usize::from(tile)]
                .clone()
                .unwrap_or_else(inactive_exchange_program);
            let active = rows[usize::from(tile)].is_some();
            let local_horizon = active
                .then(|| plan_event_cycles(&row))
                .transpose()?
                .unwrap_or(horizon);
            programs[usize::from(tile)]
                .steps
                .push(TileStep::Exchange(ExchangeStep {
                    active,
                    program: PlacedExchangeRow {
                        address,
                        words: row,
                    },
                    wait_cycles: horizon - local_horizon,
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
        .enumerate()
        .filter(|(_, bytes)| bytes.iter().any(|byte| *byte != 0))
        .map(|(tile, data)| TileProgramData {
            tile: tile as u16,
            address: DATA_BASE,
            data,
        })
        .collect::<Vec<_>>();
    let application = build_tile_program_package(&programs, &data, toolchain, runtime_source)?;
    eprintln!(
        "exchangeStress seed={seed:#x} cases={cases} transfers={} activeTiles={ACTIVE_TILES} maxWords={maximum_words} maxComputeDelay={maximum_compute_delay}",
        transfers.len()
    );
    Ok(StressPackage {
        application,
        transfers,
        row_ranges,
    })
}

impl StressPackage {
    pub(crate) fn failure_context(&self, runtime: &Runtime) -> String {
        let topology = Topology::c600();
        let mut stopped = Vec::new();
        for logical in 0..ACTIVE_TILES {
            let Ok(physical) = topology.physical(logical) else {
                continue;
            };
            let Ok(pc) = runtime.device().read_tile_program_counter(physical, 0) else {
                continue;
            };
            if let Some((case, _)) = self
                .row_ranges
                .iter()
                .enumerate()
                .find(|(_, (start, end))| (*start..*end).contains(&pc))
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
        format!("random exchange failure; stoppedRows={stopped:?}; transfers={transfers:?}")
    }
}

fn allocate(cursors: &mut [u32], tile: u16, bytes: u32, rng: &mut fastrand::Rng) -> Result<u32> {
    let cursor = &mut cursors[usize::from(tile)];
    *cursor = cursor
        .checked_add(rng.u32(0..=8) * 4)
        .context("test allocation overflow")?;
    let address = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .context("test allocation overflow")?;
    if *cursor > DATA_LIMIT {
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

fn word_value(case: u32, source: u16, destinations: &[u16], index: u32) -> u32 {
    let mut value =
        0x9e37_79b9u32 ^ case.wrapping_mul(0x85eb_ca6b) ^ index.wrapping_mul(0xc2b2_ae35);
    value ^= u32::from(source) << 16;
    for &destination in destinations {
        value = value.rotate_left(5) ^ u32::from(destination);
    }
    value
}

fn fill_source(
    buffer: &mut [u8],
    address: u32,
    case: u32,
    source: u16,
    destinations: &[u16],
    words: u32,
) {
    let offset = (address - DATA_BASE) as usize;
    for index in 0..words {
        let start = offset + index as usize * 4;
        buffer[start..start + 4]
            .copy_from_slice(&word_value(case, source, destinations, index).to_le_bytes());
    }
}
