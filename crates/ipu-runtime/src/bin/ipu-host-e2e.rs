use ipu_compiler::{Allocation, AllocationKind, Schedule, TensorId};
use ipu_elf::Toolchain;
use ipu_package::{Binding, RegionSlice};
use ipu_runtime::{ExecutableGraph, package_graph, run_host};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const TILE_COUNT: u16 = 1472;
const SOURCE_TILE: u16 = 0;
const SOURCE_ADDRESS: u32 = 0x53000;
const TRANSFER_BYTES: u32 = 64;

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
    let input = test_payload();
    let graph = host_exchange_graph();
    let app = package_graph(&graph, &[runtime_object]).unwrap();
    let result = run_host(&app, &bootloader, &configuration, &device, &input).unwrap();
    assert_eq!(result, input);
    println!("hostBytes={} h2d=PASS d2h=PASS", result.len());
    let _ = fs::remove_dir_all(output);
}

fn host_exchange_graph() -> ExecutableGraph {
    let topology = ipu_exchange::Topology::c600();
    let tensor = TensorId(0);
    ExecutableGraph {
        schedule: Schedule {
            layouts: Vec::new(),
            phases: Vec::new(),
            allocations: vec![Allocation {
                tensor,
                tile: SOURCE_TILE,
                address: SOURCE_ADDRESS,
                size: TRANSFER_BYTES,
                live_from: 0,
                live_until: usize::MAX,
                kind: AllocationKind::Home,
            }],
            tile_count: TILE_COUNT,
            peak_sram: BTreeMap::new(),
        },
        initial_buffers: Vec::new(),
        outputs: Vec::new(),
        host_inputs: vec![binding(
            "input",
            topology.physical(SOURCE_TILE).unwrap(),
            SOURCE_ADDRESS,
        )],
        host_outputs: vec![binding(
            "output",
            topology.physical(SOURCE_TILE).unwrap(),
            SOURCE_ADDRESS,
        )],
    }
}

fn binding(name: &str, physical_tile: u16, address: u32) -> Binding {
    Binding {
        name: name.into(),
        dtype: "u8".into(),
        shape: vec![TRANSFER_BYTES],
        slices: vec![RegionSlice {
            tile: u32::from(physical_tile),
            tile_address: address,
            file_offset: 0,
            size: u64::from(TRANSFER_BYTES),
        }],
    }
}

fn test_payload() -> Vec<u8> {
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
        ^ u64::from(std::process::id());
    (0..TRANSFER_BYTES)
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
