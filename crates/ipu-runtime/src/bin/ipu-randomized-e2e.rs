use ipu_compiler::{
    Allocation, AllocationKind, MemoryConstraint, MemoryPlacement, Phase, Schedule, TensorId,
    Transfer, find_free_region,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{ExecutableGraph, package_graph, run_host};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const TILE_COUNT: u16 = 1472;
const DEFAULT_SEED: u64 = 0x4950_552d_5354_4143;
const DEFAULT_CASES: usize = 8;
const WORD_COUNTS: [u32; 12] = [1, 2, 15, 16, 17, 31, 63, 64, 65, 127, 256, 1024];

fn main() {
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
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/graph_runtime.S");
    let artifact = Toolchain::from_sdk(sdk)
        .compile(&source, &output, "randomized-e2e", &[])
        .unwrap();
    let runtime_object = fs::read(artifact.object).unwrap();
    let mut rng = fastrand::Rng::with_seed(seed);

    for case in 0..case_count {
        let case_seed = rng.u64(..);
        let words = WORD_COUNTS[case % WORD_COUNTS.len()];
        let fanout = 1 + case % 4;
        for output in 0..fanout {
            let (graph, input, expected) = randomized_case(case_seed, words, fanout, output);
            let app = package_graph(&graph, std::slice::from_ref(&runtime_object)).unwrap();
            let result = run_host(&app, &bootloader, &configuration, &device, &input).unwrap();
            assert_eq!(
                result, expected,
                "seed={seed:#x} case={case} output={output}"
            );
        }
        tracing::info!(
            seed,
            case,
            case_seed,
            words,
            fanout,
            "randomized exchange case passed"
        );
        println!(
            "seed={seed:#x} case={case} caseSeed={case_seed:#x} words={words} fanout={fanout} PASS"
        );
    }
    let _ = fs::remove_dir_all(output);
}

fn randomized_case(
    seed: u64,
    words: u32,
    fanout: usize,
    output: usize,
) -> (ExecutableGraph, Vec<u8>, Vec<u8>) {
    let topology = ipu_exchange::Topology::c600();
    let mut rng = fastrand::Rng::with_seed(seed);
    let relay = random_tile_except(&mut rng, &[0]);
    let mut destinations = Vec::with_capacity(fanout);
    while destinations.len() < fanout {
        let destination = random_tile_except(&mut rng, &[0, relay]);
        if !destinations.contains(&destination) {
            destinations.push(destination);
        }
    }
    let bytes = words * 4;
    let tensor = TensorId(0);
    let payload = (0..words)
        .map(|word| payload_word(seed, word))
        .collect::<Vec<_>>();
    let input = words_to_bytes(&payload);
    let expected = input.clone();

    let host_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let exchange_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let source_address = find_free_region(&[], 0, bytes, 0, usize::MAX, host_constraint).unwrap();
    let mut allocations = vec![allocation(
        tensor,
        0,
        source_address,
        bytes,
        AllocationKind::Home,
        0,
    )];
    let relay_address = find_free_region(
        &allocations,
        relay,
        bytes,
        0,
        usize::MAX,
        exchange_constraint,
    )
    .unwrap();
    allocations.push(allocation(
        tensor,
        relay,
        relay_address,
        bytes,
        AllocationKind::ExchangeStaging { phase: 0 },
        0,
    ));
    let destination_addresses = destinations
        .iter()
        .copied()
        .map(|destination| {
            let address = find_free_region(
                &allocations,
                destination,
                bytes,
                1,
                usize::MAX,
                exchange_constraint,
            )
            .unwrap();
            allocations.push(allocation(
                tensor,
                destination,
                address,
                bytes,
                AllocationKind::ExchangeStaging { phase: 1 },
                1,
            ));
            address
        })
        .collect::<Vec<_>>();
    let phases = vec![
        Phase::Exchange {
            transfers: vec![Transfer {
                source_tile: 0,
                destination_tile: relay,
                tensor,
                bytes,
            }],
        },
        Phase::Exchange {
            transfers: destinations
                .iter()
                .copied()
                .map(|destination| Transfer {
                    source_tile: relay,
                    destination_tile: destination,
                    tensor,
                    bytes,
                })
                .collect(),
        },
    ];
    let host_inputs = vec![binding(
        "input",
        vec![(topology.physical(0).unwrap(), source_address, bytes)],
        words,
    )];
    let host_outputs = vec![binding(
        "output",
        vec![(
            topology.physical(destinations[output]).unwrap(),
            destination_addresses[output],
            bytes,
        )],
        words,
    )];
    (
        ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases,
                allocations,
                tile_count: TILE_COUNT,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers: Vec::new(),
            outputs: Vec::new(),
            host_inputs,
            host_outputs,
        },
        input,
        expected,
    )
}

fn allocation(
    tensor: TensorId,
    tile: u16,
    address: u32,
    size: u32,
    kind: AllocationKind,
    live_from: usize,
) -> Allocation {
    Allocation {
        tensor,
        tile,
        address,
        size,
        live_from,
        live_until: usize::MAX,
        kind,
    }
}

fn binding(name: &str, regions: Vec<(u16, u32, u32)>, words: u32) -> Binding {
    let mut file_offset = 0u64;
    let slices = regions
        .into_iter()
        .map(|(tile, tile_address, bytes)| {
            let slice = RegionSlice {
                tile: u32::from(tile),
                tile_address,
                file_offset,
                size: u64::from(bytes),
            };
            file_offset += u64::from(bytes);
            slice
        })
        .collect();
    Binding {
        name: name.into(),
        dtype: "u32".into(),
        shape: vec![words],
        slices,
    }
}

fn random_tile_except(rng: &mut fastrand::Rng, excluded: &[u16]) -> u16 {
    loop {
        let tile = rng.u16(0..TILE_COUNT);
        if !excluded.contains(&tile) {
            return tile;
        }
    }
}

fn payload_word(seed: u64, word: u32) -> u32 {
    (seed as u32).rotate_left(word & 31) ^ word.wrapping_mul(0x85eb_ca6b)
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
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

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set for the hardware test"))
}
