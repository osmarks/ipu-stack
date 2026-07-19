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
const ACCUMULATOR_ADDRESS: u32 = 0x60000;
const REDUCTION_STAGING_ADDRESS: u32 = 0x51040;
const PERMUTATION_SOURCE_ADDRESS: u32 = 0x60010;
const PERMUTATION_STAGING_ADDRESS: u32 = 0x54000;
const MULTICAST_SOURCE_ADDRESS: u32 = 0x61000;
const MULTICAST_STAGING_ADDRESS: u32 = 0x53000;
const MULTICAST_WORDS: usize = 64;
const PERMUTATION_MULTIPLIER: u32 = 5;
const PERMUTATION_OFFSET: u32 = 17;
const RELAY_TILE: u16 = 736;
const MULTICAST_TAP_TILE: u16 = 274;
const RELAY_DESTINATION_TILE: u16 = 1286;
const REDUCTION_PHASES: usize = 22;

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
    let output = std::env::temp_dir().join(format!("ipu-stack-e2e-{}", std::process::id()));
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S");
    let toolchain = Toolchain::from_sdk(sdk);
    let artifact = toolchain
        .compile(&source, &output, "exchange-e2e", &[])
        .unwrap();
    let runtime_object = fs::read(artifact.object).unwrap();
    let kernel_source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/add_u32.S");
    let kernel = toolchain
        .compile(&kernel_source, &output, "add-u32-e2e", &[])
        .unwrap();
    let kernel_object = fs::read(kernel.object).unwrap();

    let mode = std::env::var("IPU_GRAPH_TEST").unwrap_or_else(|_| "all".into());
    let (graph, expected_sum, expected_permutation, expected_multicast) = acceptance_case(&mode);
    let app = package_graph(&graph, &[runtime_object, kernel_object]).unwrap();
    let results = run_diagnostic(&app, &bootloader, &configuration, &device).unwrap();

    for binding in &graph.outputs {
        let actual = &results.bindings[&binding.name];
        match binding.name.as_str() {
            "sum" => assert_eq!(actual, &[expected_sum]),
            "permutation" => assert_eq!(actual, &expected_permutation),
            "multicast-tap" | "relay-destination" => {
                assert_eq!(actual, &expected_multicast)
            }
            name => panic!("acceptance graph contains unchecked output {name:?}"),
        }
    }
    assert_eq!(results.bindings["runtime-completion"], [1]);
    let _ = fs::remove_dir_all(output);
}

fn acceptance_case(mode: &str) -> (ExecutableGraph, u32, Vec<u32>, Vec<u32>) {
    let (mut graph, sum, permutation, multicast) = acceptance_graph();
    match mode {
        "all" => {}
        "reduction-permutation" => {
            graph.schedule.phases.truncate(REDUCTION_PHASES + 1);
            graph
                .schedule
                .allocations
                .retain(|allocation| allocation.tensor.0 < usize::from(TILE_COUNT) * 2);
            graph.initial_buffers.retain(|buffer| {
                matches!(
                    buffer.address,
                    ACCUMULATOR_ADDRESS | PERMUTATION_SOURCE_ADDRESS | PERMUTATION_STAGING_ADDRESS
                )
            });
            graph
                .outputs
                .retain(|binding| matches!(binding.name.as_str(), "sum" | "permutation"));
        }
        "reduction" => {
            graph.schedule.phases.truncate(REDUCTION_PHASES);
            graph
                .schedule
                .allocations
                .retain(|allocation| allocation.tensor.0 < usize::from(TILE_COUNT));
            graph
                .initial_buffers
                .retain(|buffer| buffer.address == ACCUMULATOR_ADDRESS);
            graph.outputs.retain(|binding| binding.name == "sum");
        }
        "permutation" => {
            graph.schedule.phases = vec![graph.schedule.phases[REDUCTION_PHASES].clone()];
            graph.schedule.allocations.retain_mut(|allocation| {
                let keep = (usize::from(TILE_COUNT)..usize::from(TILE_COUNT) * 2)
                    .contains(&allocation.tensor.0);
                if keep {
                    allocation.live_from = 0;
                    allocation.live_until = 1;
                    if matches!(allocation.kind, AllocationKind::ExchangeStaging { .. }) {
                        allocation.kind = AllocationKind::ExchangeStaging { phase: 0 };
                    }
                }
                keep
            });
            graph.initial_buffers.retain(|buffer| {
                matches!(
                    buffer.address,
                    PERMUTATION_SOURCE_ADDRESS | PERMUTATION_STAGING_ADDRESS
                )
            });
            graph
                .outputs
                .retain(|binding| binding.name == "permutation");
        }
        "point" | "fanout" | "multicast" => {
            graph.schedule.phases = vec![graph.schedule.phases[REDUCTION_PHASES + 1].clone()];
            if let Phase::Exchange { transfers } = &mut graph.schedule.phases[0] {
                if mode == "point" {
                    transfers.truncate(2);
                    transfers.remove(0);
                } else if mode == "fanout" {
                    transfers.last_mut().unwrap().source_tile = 0;
                }
            }
            graph.schedule.allocations.retain_mut(|allocation| {
                let keep = allocation.tensor.0 == usize::from(TILE_COUNT) * 2;
                if keep {
                    allocation.live_from = 0;
                    allocation.live_until = 1;
                    if matches!(allocation.kind, AllocationKind::ExchangeStaging { .. }) {
                        allocation.kind = AllocationKind::ExchangeStaging { phase: 0 };
                    }
                }
                keep
            });
            graph
                .initial_buffers
                .retain(|buffer| buffer.address == MULTICAST_SOURCE_ADDRESS);
            graph.outputs.retain(|binding| {
                binding.name == "multicast-tap"
                    || (mode != "point" && binding.name == "relay-destination")
            });
        }
        _ => panic!(
            "IPU_GRAPH_TEST must be reduction, reduction-permutation, permutation, point, fanout, multicast, or all"
        ),
    }
    (graph, sum, permutation, multicast)
}

fn acceptance_graph() -> (ExecutableGraph, u32, Vec<u32>, Vec<u32>) {
    let topology = ipu_exchange::Topology::c600();
    let mut phases = Vec::new();
    let mut allocations = Vec::new();
    let mut initial_buffers = Vec::new();

    for tile in 0..TILE_COUNT {
        let tensor = TensorId(usize::from(tile));
        allocations.push(home(tensor, tile, ACCUMULATOR_ADDRESS, 4));
        initial_buffers.push(InitialBuffer {
            tile,
            address: ACCUMULATOR_ADDRESS,
            words: vec![u32::from(tile) + 1],
        });
    }

    let mut active = (0..TILE_COUNT).collect::<Vec<_>>();
    let mut operation = 0usize;
    while active.len() > 1 {
        let exchange_phase = phases.len();
        let mut transfers = Vec::new();
        let mut compute = Vec::new();
        let mut next = Vec::with_capacity(active.len().div_ceil(2));
        for pair in active.chunks(2) {
            let receiver = pair[0];
            next.push(receiver);
            if let Some(&sender) = pair.get(1) {
                let sender_tensor = TensorId(usize::from(sender));
                transfers.push(Transfer {
                    source_tile: sender,
                    destination_tile: receiver,
                    tensor: sender_tensor,
                    bytes: 4,
                });
                allocations.push(Allocation {
                    tensor: sender_tensor,
                    tile: receiver,
                    address: REDUCTION_STAGING_ADDRESS,
                    size: 4,
                    live_from: exchange_phase,
                    live_until: exchange_phase + 1,
                    kind: AllocationKind::ExchangeStaging {
                        phase: exchange_phase,
                    },
                });
                compute.push(add_u32(
                    operation,
                    receiver,
                    TensorId(usize::from(receiver)),
                    sender_tensor,
                ));
                operation += 1;
            }
        }
        phases.push(Phase::Exchange { transfers });
        phases.push(Phase::Compute {
            op: OpId(operation),
            commands: compute,
        });
        active = next;
    }

    let permutation_phase = phases.len();
    let mut permutation_transfers = Vec::with_capacity(usize::from(TILE_COUNT));
    let mut permutation_expected = vec![0; usize::from(TILE_COUNT)];
    for source in 0..TILE_COUNT {
        let destination = affine_destination(source);
        let tensor = TensorId(usize::from(TILE_COUNT) + usize::from(source));
        let value = permutation_value(source);
        allocations.push(home(tensor, source, PERMUTATION_SOURCE_ADDRESS, 4));
        allocations.push(Allocation {
            tensor,
            tile: destination,
            address: PERMUTATION_STAGING_ADDRESS,
            size: 4,
            live_from: permutation_phase,
            live_until: permutation_phase + 1,
            kind: AllocationKind::ExchangeStaging {
                phase: permutation_phase,
            },
        });
        initial_buffers.push(InitialBuffer {
            tile: source,
            address: PERMUTATION_SOURCE_ADDRESS,
            words: vec![value],
        });
        initial_buffers.push(InitialBuffer {
            tile: destination,
            address: PERMUTATION_STAGING_ADDRESS,
            words: vec![0xdead_0000 | u32::from(destination)],
        });
        permutation_transfers.push(Transfer {
            source_tile: source,
            destination_tile: destination,
            tensor,
            bytes: 4,
        });
        permutation_expected[usize::from(destination)] = value;
    }
    phases.push(Phase::Exchange {
        transfers: permutation_transfers,
    });

    let multicast_phase = phases.len();
    let multicast_tensor = TensorId(usize::from(TILE_COUNT) * 2);
    let multicast_payload = (0..MULTICAST_WORDS)
        .map(|index| multicast_word(index as u32))
        .collect::<Vec<_>>();
    allocations.push(home(
        multicast_tensor,
        0,
        MULTICAST_SOURCE_ADDRESS,
        (MULTICAST_WORDS * 4) as u32,
    ));
    initial_buffers.push(InitialBuffer {
        tile: 0,
        address: MULTICAST_SOURCE_ADDRESS,
        words: multicast_payload.clone(),
    });
    for destination in [RELAY_TILE, MULTICAST_TAP_TILE, RELAY_DESTINATION_TILE] {
        allocations.push(Allocation {
            tensor: multicast_tensor,
            tile: destination,
            address: MULTICAST_STAGING_ADDRESS,
            size: (MULTICAST_WORDS * 4) as u32,
            live_from: multicast_phase,
            live_until: multicast_phase + 1,
            kind: AllocationKind::ExchangeStaging {
                phase: multicast_phase,
            },
        });
    }
    phases.push(Phase::Exchange {
        transfers: vec![
            Transfer {
                source_tile: 0,
                destination_tile: RELAY_TILE,
                tensor: multicast_tensor,
                bytes: (MULTICAST_WORDS * 4) as u32,
            },
            Transfer {
                source_tile: 0,
                destination_tile: MULTICAST_TAP_TILE,
                tensor: multicast_tensor,
                bytes: (MULTICAST_WORDS * 4) as u32,
            },
            Transfer {
                source_tile: RELAY_TILE,
                destination_tile: RELAY_DESTINATION_TILE,
                tensor: multicast_tensor,
                bytes: (MULTICAST_WORDS * 4) as u32,
            },
        ],
    });

    let outputs = vec![
        binding(
            "sum",
            vec![(
                topology.physical(active[0]).unwrap(),
                ACCUMULATOR_ADDRESS,
                4,
            )],
        ),
        binding(
            "permutation",
            (0..TILE_COUNT)
                .map(|tile| {
                    (
                        topology.physical(tile).unwrap(),
                        PERMUTATION_STAGING_ADDRESS,
                        4,
                    )
                })
                .collect(),
        ),
        binding(
            "multicast-tap",
            vec![(
                topology.physical(MULTICAST_TAP_TILE).unwrap(),
                MULTICAST_STAGING_ADDRESS,
                (MULTICAST_WORDS * 4) as u64,
            )],
        ),
        binding(
            "relay-destination",
            vec![(
                topology.physical(RELAY_DESTINATION_TILE).unwrap(),
                MULTICAST_STAGING_ADDRESS,
                (MULTICAST_WORDS * 4) as u64,
            )],
        ),
    ];
    let count = u32::from(TILE_COUNT);
    let expected_sum = count * (count + 1) / 2;
    (
        ExecutableGraph {
            schedule: Schedule {
                layouts: Vec::new(),
                phases,
                allocations,
                tile_count: TILE_COUNT,
                peak_sram: BTreeMap::new(),
            },
            initial_buffers,
            outputs,
            host_inputs: Vec::new(),
            host_outputs: Vec::new(),
        },
        expected_sum,
        permutation_expected,
        multicast_payload,
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

fn add_u32(
    operation: usize,
    tile: u16,
    accumulator: TensorId,
    received: TensorId,
) -> KernelCommand {
    KernelCommand {
        tile,
        output: accumulator,
        inputs: vec![accumulator, received],
        arguments: Vec::new(),
        specialization: SpecializationKey {
            operation: "add_u32".into(),
            shape: vec![1],
            worker_count: 1,
            role: format!("reduction-{operation}"),
            alignment: 4,
        },
        metadata: Default::default(),
    }
}

fn affine_destination(source: u16) -> u16 {
    ((u32::from(source) * PERMUTATION_MULTIPLIER + PERMUTATION_OFFSET) % u32::from(TILE_COUNT))
        as u16
}

fn permutation_value(source: u16) -> u32 {
    u32::from(source) * 1_000_003 + 97
}

fn multicast_word(index: u32) -> u32 {
    index.wrapping_mul(2_654_435_761).wrapping_add(0x1357_9bdf)
}

fn binding(name: &str, regions: Vec<(u16, u32, u64)>) -> Binding {
    let mut file_offset = 0;
    let slices = regions
        .into_iter()
        .map(|(tile, tile_address, size)| {
            let slice = RegionSlice {
                tile: u32::from(tile),
                tile_address,
                file_offset,
                size,
            };
            file_offset += size;
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
