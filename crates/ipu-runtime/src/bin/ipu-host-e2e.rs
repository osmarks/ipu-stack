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
const HOST_CONTROLLER_TILE: u16 = 0;
const EXCHANGE_RELAY_TILE: u16 = 1;
const DEFAULT_TRANSFER_BYTES: u32 = 64;

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
    let output = std::env::temp_dir().join(format!("ipu-stack-host-e2e-{}", std::process::id()));
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../device/graph_runtime.S");
    let artifact = Toolchain::from_sdk(sdk)
        .compile(&source, &output, "host-e2e", &[])
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
    let host_tile = std::env::var("IPU_HOST_TEST_TILE")
        .map(|value| {
            value
                .parse()
                .expect("IPU_HOST_TEST_TILE must be an integer")
        })
        .unwrap_or(HOST_CONTROLLER_TILE);
    let input = test_payload(transfer_bytes);
    let graph = host_exchange_graph(transfer_bytes, exchange, host_tile).unwrap();
    let app = package_graph(&graph, &[runtime_object]).unwrap();
    let result = run_host(&app, &bootloader, &configuration, &device, &input).unwrap();
    assert_eq!(result, input);
    println!(
        "hostBytes={} h2d=PASS exchange={} d2h=PASS",
        result.len(),
        if exchange { "PASS" } else { "SKIP" }
    );
    let _ = fs::remove_dir_all(output);
}

fn host_exchange_graph(
    transfer_bytes: u32,
    exchange: bool,
    host_tile: u16,
) -> Result<ExecutableGraph, ipu_compiler::CompileError> {
    let topology = ipu_exchange::Topology::c600();
    let tensor = TensorId(0);
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
    let source_address = find_free_region(
        &[],
        source_tile,
        transfer_bytes,
        0,
        source_live_until,
        host_constraint,
    )?;
    let mut allocations = vec![Allocation {
        tensor,
        tile: source_tile,
        address: source_address,
        size: transfer_bytes,
        live_from: 0,
        live_until: source_live_until,
        kind: AllocationKind::Home,
    }];
    let mut output_address = source_address;
    let phases = if exchange {
        let relay_address = find_free_region(
            &allocations,
            EXCHANGE_RELAY_TILE,
            transfer_bytes,
            0,
            2,
            exchange_constraint,
        )?;
        allocations.push(Allocation {
            tensor,
            tile: EXCHANGE_RELAY_TILE,
            address: relay_address,
            size: transfer_bytes,
            live_from: 0,
            live_until: 2,
            kind: AllocationKind::ExchangeStaging { phase: 0 },
        });
        output_address = find_free_region(
            &allocations,
            HOST_CONTROLLER_TILE,
            transfer_bytes,
            1,
            2,
            exchange_constraint,
        )?;
        allocations.push(Allocation {
            tensor,
            tile: HOST_CONTROLLER_TILE,
            address: output_address,
            size: transfer_bytes,
            live_from: 1,
            live_until: 2,
            kind: AllocationKind::ExchangeStaging { phase: 1 },
        });
        vec![
            Phase::Exchange {
                transfers: vec![Transfer {
                    source_tile: HOST_CONTROLLER_TILE,
                    destination_tile: EXCHANGE_RELAY_TILE,
                    tensor,
                    bytes: transfer_bytes,
                }],
            },
            Phase::Exchange {
                transfers: vec![Transfer {
                    source_tile: EXCHANGE_RELAY_TILE,
                    destination_tile: HOST_CONTROLLER_TILE,
                    tensor,
                    bytes: transfer_bytes,
                }],
            },
        ]
    } else {
        Vec::new()
    };
    let output_tile = if exchange {
        HOST_CONTROLLER_TILE
    } else {
        source_tile
    };
    Ok(ExecutableGraph {
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
            transfer_bytes,
        )],
        host_outputs: vec![binding(
            "output",
            topology.physical(output_tile).unwrap(),
            output_address,
            transfer_bytes,
        )],
    })
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
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
        ^ u64::from(std::process::id());
    (0..transfer_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        })
        .collect()
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}
