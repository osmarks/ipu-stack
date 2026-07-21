use std::fs;
use std::process::{Command, ExitStatus};
use std::sync::{Mutex, MutexGuard};

const RANDOM_SEEDS: [u64; 3] = [
    0x4950_552d_5354_4143,
    0x6a09_e667_f3bc_c909,
    0xbb67_ae85_84ca_a73b,
];

static DEVICE: Mutex<()> = Mutex::new(());

#[test]
fn initialized_local_tile_d2h_works() {
    let _device = device();
    require_success(
        "initialized tile D2H",
        host_test(&[("IPU_HOST_TEST_D2H_ONLY", "1")]),
    );
}

#[test]
fn local_h2d_and_d2h_round_trip_works() {
    let _device = device();
    require_success("local H2D and D2H", host_test(&[]));
}

#[test]
fn resident_h2d_then_d2h_works() {
    let _device = device();
    require_success(
        "resident H2D initialization followed by D2H",
        host_test(&[("IPU_HOST_TEST_RESIDENT_INPUT", "1")]),
    );
}

#[test]
fn initialized_d2d_relay_and_d2h_work() {
    let _device = device();
    require_success(
        "initialized D2D relay and D2H",
        host_test(&[
            ("IPU_HOST_TEST_EXCHANGE", "1"),
            ("IPU_HOST_TEST_INITIALIZED_EXCHANGE", "1"),
        ]),
    );
}

#[test]
fn large_h2d_timed_relay_and_d2h_composition_works() {
    let _device = device();
    require_success(
        "large H2D, timed D2D relay, and D2H composition",
        host_test(&[
            ("IPU_HOST_TEST_EXCHANGE", "1"),
            ("IPU_HOST_TEST_BYTES", "2048"),
        ]),
    );
}

#[test]
fn h2d_compute_d2d_compute_d2h_works() {
    let _device = device();
    require_success(
        "H2D, compute, D2D, compute, and D2H composition",
        host_test(&[
            ("IPU_HOST_TEST_COMPUTE_RELAY", "1"),
            ("IPU_HOST_TEST_BYTES", "4"),
        ]),
    );
}

#[test]
fn compiler_segmented_allocation_executes_on_hardware() {
    let _device = device();
    require_success(
        "compiler-generated add spanning segmented SRAM arenas",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "compiler-segmented-add")
            .status()
            .expect("launch compiler allocation hardware test runner"),
    );
}

#[test]
fn remote_tile_h2d_and_d2h_work() {
    let _device = device();
    require_success(
        "remote-tile H2D and D2H",
        host_test(&[
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_BYTES", "8192"),
        ]),
    );
}

#[test]
fn high_address_large_h2d_and_d2h_work() {
    let _device = device();
    require_success(
        "64-KiB high-address H2D and D2H",
        host_test(&[
            ("IPU_HOST_TEST_TILE", "100"),
            ("IPU_HOST_TEST_BYTES", "65536"),
            ("IPU_HOST_TEST_ADDRESS", "0x60000"),
        ]),
    );
}

#[test]
fn initialized_remote_tile_d2h_works() {
    let _device = device();
    require_success(
        "remote-tile initialized D2H",
        host_test(&[
            ("IPU_HOST_TEST_REMOTE_D2H", "1"),
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_BYTES", "8192"),
        ]),
    );
}

#[test]
fn distinct_sources_can_write_disjoint_host_ranges() {
    let _device = device();
    require_success(
        "distinct multi-source D2H",
        host_test(&[
            ("IPU_HOST_TEST_TILES", "1471,274"),
            ("IPU_HOST_TEST_BYTES", "64"),
        ]),
    );
}

#[test]
fn host_routes_cover_every_endpoint_field() {
    let _device = device();
    let topology = ipu_exchange::Topology::c600();
    let physical_tiles = (0..64)
        .chain((1..23).map(|row| row * 64 + 17))
        .collect::<Vec<_>>();
    let logical_tiles = physical_tiles
        .iter()
        .map(|&physical| {
            (0..1472)
                .find(|&logical| topology.physical(logical).unwrap() == physical)
                .expect("physical C600 tile must have a logical tile")
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    require_success(
        "host routes spanning every endpoint field",
        host_test(&[
            ("IPU_HOST_TEST_TILES", &logical_tiles),
            ("IPU_HOST_TEST_BYTES", "64"),
        ]),
    );
}

#[test]
fn randomized_multi_tile_host_round_trips_work() {
    let _device = device();
    for (seed, bytes) in [
        (0x243f_6a88_85a3_08d3, 4u32),
        (0x1319_8a2e_0370_7344, 1024),
        (0xa409_3822_299f_31d0, 8192),
    ] {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut tiles = std::collections::BTreeSet::new();
        while tiles.len() < 8 {
            tiles.insert(rng.u16(0..1472));
        }
        let tile_list = tiles
            .iter()
            .map(u16::to_string)
            .collect::<Vec<_>>()
            .join(",");
        require_success(
            &format!("randomized host round trip seed={seed:#x} bytes={bytes} tiles={tile_list}"),
            host_test(&[
                ("IPU_HOST_TEST_TILES", &tile_list),
                ("IPU_HOST_TEST_BYTES", &bytes.to_string()),
            ]),
        );
    }
}

#[test]
fn all_tile_reduction_works() {
    let _device = device();
    require_success(
        "all-tile reduction graph",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "reduction")
            .status()
            .expect("launch reduction hardware test runner"),
    );
}

#[test]
fn sparse_compute_to_dense_exchange_works() {
    let _device = device();
    require_success(
        "sparse compute followed by an all-tile exchange",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "reduction-permutation")
            .status()
            .expect("launch compute-to-exchange hardware test runner"),
    );
}

#[test]
fn all_tile_permutation_works() {
    let _device = device();
    require_success(
        "all-tile permutation graph",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "permutation")
            .status()
            .expect("launch permutation hardware test runner"),
    );
}

#[test]
fn direct_multicast_works() {
    let _device = device();
    require_success(
        "single-source multicast graph",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "fanout")
            .status()
            .expect("launch direct multicast hardware test runner"),
    );
}

#[test]
fn multicast_receiver_can_send_in_same_phase() {
    let _device = device();
    require_success(
        "multicast graph with a receiver that sends in the same phase",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .env("IPU_GRAPH_TEST", "multicast")
            .status()
            .expect("launch multicast hardware test runner"),
    );
}

#[test]
fn randomized_static_d2d_graphs_work() {
    let _device = device();
    for seed in RANDOM_SEEDS {
        require_success(
            &format!("randomized exchange graph seed {seed:#x}"),
            Command::new(env!("CARGO_BIN_EXE_ipu-randomized-e2e"))
                .env("IPU_RANDOM_SEED", seed.to_string())
                .status()
                .expect("launch randomized hardware test runner"),
        );
    }
}

#[test]
fn blocked_gemm_with_cycle_profile_works() {
    let _device = device();
    let path = std::env::temp_dir().join(format!(
        "ipu-stack-gemm-profile-{}.capnp",
        std::process::id()
    ));
    let mut command = Command::new(env!("CARGO_BIN_EXE_ipu-gemm-e2e"));
    command
        .env("IPU_GEMM_DIMENSION", "128")
        .env("IPU_PROFILE_OUTPUT", &path)
        .env("IPU_GEMM_SRAM_CHECK_BLOCK", "1,1");
    configure_host_write_jitter(&mut command);
    let status = command
        .status()
        .expect("launch profiled blocked GEMM hardware test runner");
    require_success("blocked GEMM with per-tile cycle profile", status);

    let profile = ipu_package::ProfileReport::read(fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(profile.tiles.len(), 1472);
    assert!(profile.clock_hz > 0);
    assert!(profile.tiles.iter().all(|tile| !tile.samples.is_empty()));
    assert!(profile.tiles.iter().any(|tile| {
        tile.samples
            .iter()
            .any(|sample| sample.step.kind == ipu_package::ProfileStepKind::Exchange)
    }));
    assert!(profile.tiles.iter().any(|tile| {
        tile.samples
            .iter()
            .any(|sample| sample.step.kind == ipu_package::ProfileStepKind::Compute)
    }));
    assert!(profile.tiles.iter().any(|tile| {
        tile.samples
            .iter()
            .any(|sample| sample.step.kind == ipu_package::ProfileStepKind::Synchronization)
    }));
    assert!(profile.tiles.iter().any(|tile| {
        tile.samples
            .iter()
            .any(|sample| sample.step.kind == ipu_package::ProfileStepKind::Idle)
    }));
    fs::remove_file(path).unwrap();
}

#[test]
fn flash_attention_matches_host_softmax_across_encoder_shapes() {
    let _device = device();
    require_success(
        "FP16 FlashAttention for encoder head dimensions and batch sizes",
        Command::new(env!("CARGO_BIN_EXE_ipu-attention-f16-e2e"))
            .status()
            .expect("launch FlashAttention hardware test runner"),
    );
}

fn device() -> MutexGuard<'static, ()> {
    DEVICE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn host_test(environment: &[(&str, &str)]) -> ExitStatus {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"));
    command.envs(environment.iter().copied());
    configure_host_write_jitter(&mut command);
    command.status().expect("launch host exchange test runner")
}

fn configure_host_write_jitter(command: &mut Command) {
    if let Some(max_delay) = std::env::var_os("IPU_TEST_HOST_WRITE_JITTER_MAX_US") {
        command.env("IPU_HOST_WRITE_JITTER_MAX_US", max_delay);
    }
    if let Some(seed) = std::env::var_os("IPU_TEST_HOST_WRITE_JITTER_SEED") {
        command.env("IPU_HOST_WRITE_JITTER_SEED", seed);
    }
}

fn require_success(capability: &str, status: ExitStatus) {
    assert!(
        status.success(),
        "hardware capability {capability:?} failed with {status}"
    );
}
