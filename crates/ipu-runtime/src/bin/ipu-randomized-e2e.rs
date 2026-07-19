use ipu_compiler::{
    Allocation, AllocationKind, KernelCommand, OpId, Phase, Schedule, SpecializationKey, TensorId,
    Transfer,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{ExecutableGraph, InitialBuffer, package_graph, run_diagnostic};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const TILE_COUNT: u16 = 1472;
const DEFAULT_SEED: u64 = 0x4950_552d_5354_4143;
const DEFAULT_CASES: usize = 6;
const PAYLOAD_WORD_COUNTS: [u32; 10] = [1, 15, 16, 17, 52, 64, 65, 127, 512, 1024];
const SOURCE_ADDRESS: u32 = 0x60000;
const FIRST_STAGING_ADDRESS: u32 = 0x52000;
const ACCUMULATOR_ADDRESS: u32 = 0x61000;
const MATCHING_SOURCE_ADDRESS: u32 = 0x61020;
const MATCHING_STAGING_ADDRESS: u32 = 0x54000;

fn main() {
    ipu_runtime::init_tracing();
    let sdk = PathBuf::from(required_env("POPLAR_SDK_ENABLED"));
    let configuration = fs::read(required_env("IPU_CONFIG")).unwrap();
    let bootloader = fs::read(
        std::env::var_os("IPU_BOOTLOADER")
            .map(PathBuf::from)
            .unwrap_or_else(|| sdk.join("bin/ipu/tile_bootloader_ipu2.elf")),
    )
    .unwrap();
    let device = std::env::var("IPU_DEVICE").unwrap_or_else(|_| "/dev/ipu0".into());
    let seed = env_number("IPU_RANDOM_SEED", DEFAULT_SEED);
    let case_count = env_number("IPU_RANDOM_CASES", DEFAULT_CASES as u64) as usize;
    assert!(case_count != 0, "IPU_RANDOM_CASES must be nonzero");

    let output =
        std::env::temp_dir().join(format!("ipu-stack-randomized-e2e-{}", std::process::id()));
    let toolchain = Toolchain::from_sdk(sdk);
    let runtime_source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S");
    let runtime = toolchain
        .compile(&runtime_source, &output, "randomized-runtime", &[])
        .unwrap();
    let kernel_source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/add_u32.S");
    let kernel = toolchain
        .compile(&kernel_source, &output, "randomized-add-u32", &[])
        .unwrap();
    let objects = [
        fs::read(runtime.object).unwrap(),
        fs::read(kernel.object).unwrap(),
    ];

    let mut seeds = fastrand::Rng::with_seed(seed);
    for case in 0..case_count {
        let case_seed = seeds.u64(..);
        let (graph, expected) = randomized_case(case_seed, case);
        let app = package_graph(&graph, &objects).unwrap();
        let actual = run_diagnostic(&app, &bootloader, &configuration, &device).unwrap();
        for (name, words) in expected {
            assert_eq!(actual.bindings[&name], words, "seed={seed:#x} case={case}");
        }
        assert_eq!(actual.bindings["runtime-completion"], [1]);
        tracing::info!(seed, case, case_seed, "randomized static D2D case passed");
    }
    let _ = fs::remove_dir_all(output);
}

fn randomized_case(seed: u64, case: usize) -> (ExecutableGraph, BTreeMap<String, Vec<u32>>) {
    let topology = ipu_exchange::Topology::c600();
    let mut rng = fastrand::Rng::with_seed(seed);
    let source = if case.is_multiple_of(3) {
        0
    } else {
        rng.u16(0..TILE_COUNT)
    };
    let fanout = rng.usize(1..=6);
    let payload_words = PAYLOAD_WORD_COUNTS[rng.usize(..PAYLOAD_WORD_COUNTS.len())];
    let payload = (0..payload_words).map(|_| rng.u32(..)).collect::<Vec<_>>();
    let payload_bytes = payload_words * 4;
    let payload_tensor = TensorId(0);

    let mut candidates = (0..TILE_COUNT)
        .filter(|tile| *tile != source)
        .collect::<Vec<_>>();
    rng.shuffle(&mut candidates);
    let receivers = candidates[..fanout].to_vec();
    let compute_tile = receivers[0];
    let accumulator_tensor = TensorId(1);
    let accumulator = rng.u32(..);

    let mut allocations = vec![
        home(payload_tensor, source, SOURCE_ADDRESS, payload_bytes),
        home(accumulator_tensor, compute_tile, ACCUMULATOR_ADDRESS, 4),
    ];
    let mut initial_buffers = vec![
        InitialBuffer {
            tile: source,
            address: SOURCE_ADDRESS,
            words: payload.clone(),
        },
        InitialBuffer {
            tile: compute_tile,
            address: ACCUMULATOR_ADDRESS,
            words: vec![accumulator],
        },
    ];
    for &receiver in &receivers {
        allocations.push(staging(
            payload_tensor,
            receiver,
            FIRST_STAGING_ADDRESS,
            payload_bytes,
            0,
            1,
        ));
    }

    let first_exchange = Phase::Exchange {
        transfers: receivers
            .iter()
            .copied()
            .map(|destination_tile| Transfer {
                source_tile: source,
                destination_tile,
                tensor: payload_tensor,
                bytes: payload_bytes,
            })
            .collect(),
    };
    let sparse_compute = Phase::Compute {
        op: OpId(0),
        commands: vec![KernelCommand {
            tile: compute_tile,
            output: accumulator_tensor,
            inputs: vec![accumulator_tensor, payload_tensor],
            arguments: Vec::new(),
            specialization: SpecializationKey {
                operation: "add_u32".into(),
                shape: vec![1],
                worker_count: 1,
                role: "randomized-sparse-compute".into(),
                alignment: 4,
            },
        }],
    };

    let matching_width = rng.usize(8..=64);
    let mut matching_tiles = (0..TILE_COUNT)
        .filter(|tile| *tile != compute_tile)
        .collect::<Vec<_>>();
    rng.shuffle(&mut matching_tiles);
    let mut matching_sources = Vec::with_capacity(matching_width);
    matching_sources.push(compute_tile);
    matching_sources.extend_from_slice(&matching_tiles[..matching_width - 1]);
    let matching_destinations = &matching_tiles[matching_width - 1..2 * matching_width - 1];
    let computed = accumulator.wrapping_add(payload[0]);
    let mut matching_values = Vec::with_capacity(matching_width);
    matching_values.push(computed);
    matching_values.extend((1..matching_width).map(|_| rng.u32(..)));

    let mut matching_transfers = Vec::with_capacity(matching_width);
    let mut output_slices = Vec::with_capacity(matching_width);
    for index in 0..matching_width {
        let tensor = if index == 0 {
            accumulator_tensor
        } else {
            TensorId(index + 1)
        };
        let source_tile = matching_sources[index];
        let destination_tile = matching_destinations[index];
        if index != 0 {
            allocations.push(home(tensor, source_tile, MATCHING_SOURCE_ADDRESS, 4));
            initial_buffers.push(InitialBuffer {
                tile: source_tile,
                address: MATCHING_SOURCE_ADDRESS,
                words: vec![matching_values[index]],
            });
        }
        allocations.push(staging(
            tensor,
            destination_tile,
            MATCHING_STAGING_ADDRESS,
            4,
            2,
            3,
        ));
        matching_transfers.push(Transfer {
            source_tile,
            destination_tile,
            tensor,
            bytes: 4,
        });
        output_slices.push((
            topology.physical(destination_tile).unwrap(),
            MATCHING_STAGING_ADDRESS,
            4,
        ));
    }

    let mut outputs = vec![binding("matching", output_slices)];
    let mut expected = BTreeMap::from([("matching".into(), matching_values)]);
    for (index, &receiver) in receivers.iter().enumerate() {
        let name = format!("multicast-{index}");
        outputs.push(binding(
            &name,
            vec![(
                topology.physical(receiver).unwrap(),
                FIRST_STAGING_ADDRESS,
                payload_bytes,
            )],
        ));
        expected.insert(name, payload.clone());
    }

    (
        ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases: vec![
                    first_exchange,
                    sparse_compute,
                    Phase::Exchange {
                        transfers: matching_transfers,
                    },
                ],
                allocations,
                tile_count: TILE_COUNT,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers,
            outputs,
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        },
        expected,
    )
}

fn home(tensor: TensorId, tile: u16, address: u32, size: u32) -> Allocation {
    Allocation {
        tensor,
        tile,
        address,
        size,
        live_from: 0,
        live_until: usize::MAX,
        kind: AllocationKind::Home,
    }
}

fn staging(
    tensor: TensorId,
    tile: u16,
    address: u32,
    size: u32,
    phase: usize,
    live_until: usize,
) -> Allocation {
    Allocation {
        tensor,
        tile,
        address,
        size,
        live_from: phase,
        live_until,
        kind: AllocationKind::ExchangeStaging { phase },
    }
}

fn binding(name: &str, regions: Vec<(u16, u32, u32)>) -> Binding {
    let mut file_offset = 0u64;
    let slices = regions
        .into_iter()
        .map(|(tile, tile_address, size)| {
            let slice = RegionSlice {
                tile: u32::from(tile),
                tile_address,
                file_offset,
                size: u64::from(size),
            };
            file_offset += u64::from(size);
            slice
        })
        .collect();
    Binding {
        name: name.into(),
        dtype: "u32".into(),
        shape: vec![(file_offset / 4) as u32],
        slices,
    }
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set for the hardware test"))
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .map(|value| {
            if let Some(hex) = value.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("{name} must be an integer"))
            } else {
                value
                    .parse()
                    .unwrap_or_else(|_| panic!("{name} must be an integer"))
            }
        })
        .unwrap_or(default)
}
