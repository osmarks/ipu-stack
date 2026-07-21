use ipu_compiler::{
    Allocation, AllocationKind, KernelCommand, MemoryConstraint, MemoryPlacement, OpId, Phase,
    Schedule, SpecializationKey, TensorId, Transfer, find_free_region,
};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{
    ExecutableGraph, HostRunOptions, InitialBuffer, package_graph, run_host_with_options,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const TILE_COUNT: u16 = 1472;
const HOST_CONTROLLER_TILE: u16 = 0;
const EXCHANGE_RELAY_TILE: u16 = 1;
const DEFAULT_TRANSFER_BYTES: u32 = 64;

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
    let output = std::env::temp_dir().join(format!("ipu-stack-host-e2e-{}", std::process::id()));
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/static_runtime.S");
    let mut compile_flags = Vec::new();
    for (environment, define) in [
        ("IPU_STACK_TRACE_MILESTONES", "IPU_STACK_TRACE_MILESTONES"),
        (
            "IPU_STACK_TRAP_AFTER_RECEIVE",
            "IPU_STACK_TRAP_AFTER_RECEIVE",
        ),
    ] {
        if std::env::var_os(environment).is_some() {
            compile_flags.push(format!("-D{define}"));
        }
    }
    let toolchain = Toolchain::from_sdk(sdk);
    let artifact = toolchain
        .compile(&source, &output, "host-e2e", &compile_flags)
        .unwrap();
    let runtime_object = fs::read(artifact.object).unwrap();
    let transfer_bytes = std::env::var("IPU_HOST_TEST_BYTES")
        .map(|value| {
            value
                .parse()
                .expect("IPU_HOST_TEST_BYTES must be an integer")
        })
        .unwrap_or(DEFAULT_TRANSFER_BYTES);
    let exchange = std::env::var_os("IPU_HOST_TEST_EXCHANGE").is_some();
    let remote_d2h = std::env::var_os("IPU_HOST_TEST_REMOTE_D2H").is_some();
    let output_count = std::env::var("IPU_HOST_TEST_OUTPUTS")
        .map(|value| {
            value
                .parse::<u32>()
                .expect("IPU_HOST_TEST_OUTPUTS must be an integer")
        })
        .unwrap_or(1);
    let host_tiles = std::env::var("IPU_HOST_TEST_TILES")
        .map(|value| {
            value
                .split(',')
                .map(|tile| {
                    tile.parse::<u16>()
                        .expect("IPU_HOST_TEST_TILES must contain integers")
                })
                .collect::<Vec<_>>()
        })
        .ok();
    let d2h_only = std::env::var_os("IPU_HOST_TEST_D2H_ONLY").is_some();
    let initialized_exchange = std::env::var_os("IPU_HOST_TEST_INITIALIZED_EXCHANGE").is_some();
    let compute_relay = std::env::var_os("IPU_HOST_TEST_COMPUTE_RELAY").is_some();
    assert!(output_count != 0, "IPU_HOST_TEST_OUTPUTS must be nonzero");
    let host_tile = std::env::var("IPU_HOST_TEST_TILE")
        .map(|value| {
            value
                .parse()
                .expect("IPU_HOST_TEST_TILE must be an integer")
        })
        .unwrap_or(if remote_d2h {
            EXCHANGE_RELAY_TILE
        } else {
            HOST_CONTROLLER_TILE
        });
    assert!(
        output_count == 1 || (!exchange && !remote_d2h),
        "multiple output slices are only supported by the direct host test"
    );
    let payload_count = host_tiles
        .as_ref()
        .map_or(output_count, |tiles| tiles.len() as u32);
    let payload = test_payload(transfer_bytes * payload_count);
    let (mut graph, input, expected) = if compute_relay {
        assert_eq!(transfer_bytes, 4, "compute relay operates on one u32");
        assert!(host_tiles.is_none());
        let input = payload.clone();
        let value = u32::from_le_bytes(input[..4].try_into().unwrap());
        (
            host_compute_relay_graph().unwrap(),
            input,
            value.wrapping_mul(4).to_le_bytes().to_vec(),
        )
    } else if d2h_only {
        assert!(host_tiles.is_none());
        (
            d2h_only_graph(transfer_bytes, host_tile, &payload).unwrap(),
            Vec::new(),
            payload,
        )
    } else if initialized_exchange {
        assert!(exchange);
        assert!(host_tiles.is_none());
        let mut graph =
            host_exchange_graph(transfer_bytes, true, host_tile, remote_d2h, output_count).unwrap();
        let source = graph.host_inputs[0].slices[0].clone();
        graph.host_inputs.clear();
        graph.initial_buffers.push(InitialBuffer {
            tile: HOST_CONTROLLER_TILE,
            address: source.tile_address,
            words: payload
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect(),
        });
        (graph, Vec::new(), payload)
    } else if let Some(host_tiles) = host_tiles {
        let graph = multi_tile_host_graph(transfer_bytes, &host_tiles).unwrap();
        (graph, payload.clone(), payload)
    } else {
        let graph = host_exchange_graph(
            transfer_bytes,
            exchange || remote_d2h,
            host_tile,
            remote_d2h,
            output_count,
        )
        .unwrap();
        (graph, payload.clone(), payload)
    };
    if let Some(address) = host_test_address() {
        relocate_direct_host_graph(&mut graph, address);
    }
    if std::env::var_os("IPU_HOST_TEST_RESIDENT_INPUT").is_some() {
        graph.host_weights.append(&mut graph.host_inputs);
    }
    if exchange
        && (std::env::var_os("IPU_STACK_TRAP_AFTER_RECEIVE").is_some()
            || std::env::var_os("IPU_STACK_TRACE_MILESTONES").is_some())
    {
        let source = graph
            .schedule
            .allocations
            .iter()
            .find(|allocation| {
                allocation.tile == HOST_CONTROLLER_TILE && allocation.kind == AllocationKind::Home
            })
            .unwrap();
        graph.outputs.push(binding(
            "diagnostic-source",
            ipu_exchange::Topology::c600()
                .physical(HOST_CONTROLLER_TILE)
                .unwrap(),
            source.address,
            transfer_bytes,
        ));
        let relay = graph
            .schedule
            .allocations
            .iter()
            .find(|allocation| {
                allocation.tile == EXCHANGE_RELAY_TILE
                    && allocation.kind == AllocationKind::ExchangeStaging { phase: 0 }
            })
            .unwrap();
        graph.outputs.push(binding(
            "diagnostic-relay",
            ipu_exchange::Topology::c600()
                .physical(EXCHANGE_RELAY_TILE)
                .unwrap(),
            relay.address,
            transfer_bytes,
        ));
    }
    let mut objects = vec![runtime_object];
    if compute_relay {
        let kernel_source =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/add_u32.S");
        let kernel = toolchain
            .compile(&kernel_source, &output, "host-compute-relay", &[])
            .unwrap();
        objects.push(fs::read(kernel.object).unwrap());
    }
    let app = package_graph(&graph, &objects).unwrap();
    if let Some(path) = std::env::var_os("IPU_HOST_TEST_PACKAGE") {
        app.write(fs::File::create(path).unwrap()).unwrap();
    }
    if d2h_only {
        let source = &graph.host_outputs[0].slices[0];
        let tile = app
            .tiles
            .iter()
            .find(|tile| tile.physical_tile == source.tile)
            .unwrap();
        let segment = tile
            .segments
            .iter()
            .find(|segment| segment.address == source.tile_address)
            .unwrap();
        let blob = &app.blobs[segment.blob].bytes;
        let start = segment.blob_offset as usize;
        assert_eq!(&blob[start..start + expected.len()], expected);
    }
    let result = run_host_with_options(
        &app,
        &bootloader,
        &configuration,
        &device,
        &input,
        HostRunOptions::from_environment().unwrap(),
    )
    .unwrap();
    assert_transfer_eq(&result, &expected);
    println!(
        "hostBytes={} h2d=PASS exchange={} d2h=PASS",
        result.len(),
        if exchange || remote_d2h || compute_relay {
            "PASS"
        } else {
            "SKIP"
        }
    );
    let _ = fs::remove_dir_all(output);
}

fn host_compute_relay_graph() -> Result<ExecutableGraph, ipu_compiler::CompileError> {
    const SOURCE_TILE: u16 = 1471;
    const DESTINATION_TILE: u16 = 274;
    const BYTES: u32 = 4;

    let topology = ipu_exchange::Topology::c600();
    let host_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let exchange_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let data_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE + 2 * ipu_exchange::EXCHANGE_WINDOW_BYTES,
        limit: ipu_package::TILE_MEMORY_BASE + ipu_package::TILE_MEMORY_SIZE,
        alignment: 32,
        placement: MemoryPlacement::Low,
    };
    let input = TensorId(0);
    let first_result = TensorId(1);
    let final_result = TensorId(2);
    let mut allocations = Vec::new();
    let input_address = find_free_region(
        &allocations,
        SOURCE_TILE,
        BYTES,
        0,
        usize::MAX,
        host_constraint,
    )?;
    allocations.push(home(input, SOURCE_TILE, input_address, BYTES));
    let first_result_address = find_free_region(
        &allocations,
        SOURCE_TILE,
        BYTES,
        0,
        usize::MAX,
        data_constraint,
    )?;
    allocations.push(home(first_result, SOURCE_TILE, first_result_address, BYTES));
    let staging_address = find_free_region(
        &allocations,
        DESTINATION_TILE,
        BYTES,
        1,
        2,
        exchange_constraint,
    )?;
    allocations.push(Allocation {
        tensor: first_result,
        tile: DESTINATION_TILE,
        address: staging_address,
        size: BYTES,
        live_from: 1,
        live_until: 2,
        kind: AllocationKind::ExchangeStaging { phase: 1 },
    });
    let final_result_address = find_free_region(
        &allocations,
        DESTINATION_TILE,
        BYTES,
        0,
        usize::MAX,
        data_constraint,
    )?;
    allocations.push(home(
        final_result,
        DESTINATION_TILE,
        final_result_address,
        BYTES,
    ));

    Ok(ExecutableGraph {
        host_weights: Vec::new(),
        schedule: Schedule {
            layouts: Vec::new(),
            phases: vec![
                Phase::Compute {
                    op: OpId(0),
                    commands: vec![add_u32(SOURCE_TILE, first_result, input, "before-exchange")],
                },
                Phase::Exchange {
                    transfers: vec![Transfer {
                        source_tile: SOURCE_TILE,
                        destination_tile: DESTINATION_TILE,
                        tensor: first_result,
                        bytes: BYTES,
                    }],
                },
                Phase::Compute {
                    op: OpId(1),
                    commands: vec![add_u32(
                        DESTINATION_TILE,
                        final_result,
                        first_result,
                        "after-exchange",
                    )],
                },
            ],
            allocations,
            tile_count: TILE_COUNT,
            peak_sram: BTreeMap::new(),
        },
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![binding(
            "input",
            topology.physical(SOURCE_TILE)?,
            input_address,
            BYTES,
        )],
        host_outputs: vec![binding(
            "output",
            topology.physical(DESTINATION_TILE)?,
            final_result_address,
            BYTES,
        )],
    })
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

fn add_u32(tile: u16, output: TensorId, input: TensorId, role: &'static str) -> KernelCommand {
    KernelCommand {
        tile,
        output,
        inputs: vec![input, input],
        arguments: Vec::new(),
        specialization: SpecializationKey {
            operation: "add_u32".into(),
            shape: vec![1],
            worker_count: 1,
            role: role.into(),
            alignment: 4,
        },
        metadata: Default::default(),
    }
}

fn relocate_direct_host_graph(graph: &mut ExecutableGraph, address: u32) {
    assert!(
        graph.schedule.phases.is_empty(),
        "IPU_HOST_TEST_ADDRESS requires a direct host graph"
    );
    assert_eq!(
        graph.schedule.allocations.len(),
        1,
        "IPU_HOST_TEST_ADDRESS requires one allocation"
    );
    let old_address = graph.schedule.allocations[0].address;
    graph.schedule.allocations[0].address = address;
    for slice in graph
        .host_inputs
        .iter_mut()
        .chain(&mut graph.host_outputs)
        .flat_map(|binding| &mut binding.slices)
    {
        slice.tile_address = address + (slice.tile_address - old_address);
    }
}

fn host_test_address() -> Option<u32> {
    std::env::var("IPU_HOST_TEST_ADDRESS").ok().map(|address| {
        u32::from_str_radix(address.trim_start_matches("0x"), 16)
            .expect("IPU_HOST_TEST_ADDRESS must be hexadecimal")
    })
}

fn assert_transfer_eq(actual: &[u8], expected: &[u8]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "host transfer length mismatch"
    );
    if let Some((offset, (&actual, &expected))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        panic!(
            "host transfer mismatch at byte {offset}: actual={actual:#04x}, expected={expected:#04x}"
        );
    }
}

fn d2h_only_graph(
    transfer_bytes: u32,
    tile: u16,
    payload: &[u8],
) -> Result<ExecutableGraph, ipu_compiler::CompileError> {
    let topology = ipu_exchange::Topology::c600();
    let tensor = TensorId(0);
    let address = ipu_exchange::EXCHANGE_WINDOW_BASE + 0x3000;
    let words = payload
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    Ok(ExecutableGraph {
        host_weights: Vec::new(),
        schedule: Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: vec![Allocation {
                tensor,
                tile,
                address,
                size: transfer_bytes,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            }],
            tile_count: TILE_COUNT,
            peak_sram: BTreeMap::new(),
        },
        initial_buffers: vec![InitialBuffer {
            tile,
            address,
            words,
        }],
        outputs: Vec::new(),
        host_inputs: Vec::new(),
        host_outputs: vec![Binding {
            name: "output".into(),
            dtype: "u8".into(),
            shape: vec![transfer_bytes],
            slices: vec![RegionSlice {
                tile: u32::from(topology.physical(tile)?),
                tile_address: address,
                file_offset: 0,
                size: u64::from(transfer_bytes),
            }],
        }],
    })
}

fn multi_tile_host_graph(
    transfer_bytes: u32,
    tiles: &[u16],
) -> Result<ExecutableGraph, ipu_compiler::CompileError> {
    assert!(!tiles.is_empty());
    assert_eq!(
        tiles
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        tiles.len(),
        "IPU_HOST_TEST_TILES must not contain duplicates"
    );
    let topology = ipu_exchange::Topology::c600();
    let constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let mut allocations = Vec::new();
    let mut slices = Vec::new();
    for (index, &tile) in tiles.iter().enumerate() {
        let address = find_free_region(
            &allocations,
            tile,
            transfer_bytes,
            0,
            usize::MAX,
            constraint,
        )?;
        allocations.push(Allocation {
            tensor: TensorId(index),
            tile,
            address,
            size: transfer_bytes,
            live_from: 0,
            live_until: usize::MAX,
            kind: AllocationKind::Home,
        });
        slices.push(RegionSlice {
            tile: u32::from(topology.physical(tile).unwrap()),
            tile_address: address,
            file_offset: index as u64 * u64::from(transfer_bytes),
            size: u64::from(transfer_bytes),
        });
    }
    let input_binding = Binding {
        name: "input".into(),
        dtype: "u8".into(),
        shape: vec![transfer_bytes * tiles.len() as u32],
        slices,
    };
    Ok(ExecutableGraph {
        host_weights: Vec::new(),
        schedule: Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations,
            tile_count: TILE_COUNT,
            peak_sram: BTreeMap::new(),
        },
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![input_binding.clone()],
        host_outputs: vec![Binding {
            name: "output".into(),
            ..input_binding
        }],
    })
}

fn host_exchange_graph(
    transfer_bytes: u32,
    exchange: bool,
    host_tile: u16,
    remote_d2h: bool,
    output_count: u32,
) -> Result<ExecutableGraph, ipu_compiler::CompileError> {
    let topology = ipu_exchange::Topology::c600();
    let tensor = TensorId(0);
    let relay_tile = if remote_d2h {
        host_tile
    } else {
        EXCHANGE_RELAY_TILE
    };
    let source_tile = if exchange {
        HOST_CONTROLLER_TILE
    } else {
        host_tile
    };
    let host_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::HOST_TO_TILE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let exchange_constraint = MemoryConstraint {
        base: ipu_exchange::EXCHANGE_WINDOW_BASE,
        limit: ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES,
        alignment: 32,
        placement: MemoryPlacement::High,
    };
    let source_live_until = if exchange { 1 } else { usize::MAX };
    let input_bytes = transfer_bytes * output_count;
    let source_address = if !exchange {
        if let Some(address) = host_test_address() {
            address
        } else {
            find_free_region(
                &[],
                source_tile,
                input_bytes,
                0,
                source_live_until,
                host_constraint,
            )?
        }
    } else {
        find_free_region(
            &[],
            source_tile,
            input_bytes,
            0,
            source_live_until,
            host_constraint,
        )?
    };
    let mut allocations = vec![Allocation {
        tensor,
        tile: source_tile,
        address: source_address,
        size: input_bytes,
        live_from: 0,
        live_until: source_live_until,
        kind: AllocationKind::Home,
    }];
    let mut output_address = source_address;
    let phases = if exchange {
        let relay_live_until = 1;
        let relay_address = find_free_region(
            &allocations,
            relay_tile,
            transfer_bytes,
            0,
            relay_live_until,
            exchange_constraint,
        )?;
        allocations.push(Allocation {
            tensor,
            tile: relay_tile,
            address: relay_address,
            size: transfer_bytes,
            live_from: 0,
            live_until: relay_live_until,
            kind: AllocationKind::ExchangeStaging { phase: 0 },
        });
        let mut transfers = vec![Transfer {
            source_tile: HOST_CONTROLLER_TILE,
            destination_tile: relay_tile,
            tensor,
            bytes: transfer_bytes,
        }];
        if remote_d2h {
            output_address = relay_address;
        } else {
            output_address = find_free_region(
                &allocations,
                HOST_CONTROLLER_TILE,
                transfer_bytes,
                0,
                1,
                exchange_constraint,
            )?;
            allocations.push(Allocation {
                tensor,
                tile: HOST_CONTROLLER_TILE,
                address: output_address,
                size: transfer_bytes,
                live_from: 0,
                live_until: 1,
                kind: AllocationKind::ExchangeStaging { phase: 0 },
            });
            transfers.push(Transfer {
                source_tile: relay_tile,
                destination_tile: HOST_CONTROLLER_TILE,
                tensor,
                bytes: transfer_bytes,
            });
        }
        vec![Phase::Exchange { transfers }]
    } else {
        Vec::new()
    };
    let output_tile = if remote_d2h {
        relay_tile
    } else if exchange {
        HOST_CONTROLLER_TILE
    } else {
        source_tile
    };
    Ok(ExecutableGraph {
        host_weights: Vec::new(),
        schedule: Schedule {
            layouts: Vec::new(),
            phases,
            allocations,
            tile_count: TILE_COUNT,
            peak_sram: BTreeMap::new(),
        },
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![binding(
            "input",
            topology.physical(source_tile).unwrap(),
            source_address,
            input_bytes,
        )],
        host_outputs: vec![repeated_binding(
            "output",
            topology.physical(output_tile).unwrap(),
            output_address,
            transfer_bytes,
            output_count,
        )],
    })
}

fn repeated_binding(
    name: &str,
    physical_tile: u16,
    address: u32,
    transfer_bytes: u32,
    count: u32,
) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u8".into(),
        shape: vec![transfer_bytes * count],
        slices: (0..count)
            .map(|index| RegionSlice {
                tile: u32::from(physical_tile),
                tile_address: address + index * transfer_bytes,
                file_offset: u64::from(index * transfer_bytes),
                size: u64::from(transfer_bytes),
            })
            .collect(),
    }
}

fn binding(name: &str, physical_tile: u16, address: u32, transfer_bytes: u32) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u8".into(),
        shape: vec![transfer_bytes],
        slices: vec![RegionSlice {
            tile: u32::from(physical_tile),
            tile_address: address,
            file_offset: 0,
            size: u64::from(transfer_bytes),
        }],
    }
}

fn test_payload(transfer_bytes: u32) -> Vec<u8> {
    let mut rng = fastrand::Rng::new();
    (0..transfer_bytes).map(|_| rng.u8(..)).collect()
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
