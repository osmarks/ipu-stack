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
fn large_multi_epoch_host_and_device_transfers_work() {
    let _device = device();
    require_success(
        "multi-epoch H2D, D2D, and D2H",
        host_test(&[
            ("IPU_HOST_TEST_EXCHANGE", "1"),
            ("IPU_HOST_TEST_BYTES", "2048"),
        ]),
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
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_SECOND_TILE", "274"),
            ("IPU_HOST_TEST_BYTES", "64"),
        ]),
    );
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
fn multicast_with_relay_role_works() {
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
fn randomized_host_and_exchange_graphs_work() {
    let _device = device();
    for seed in RANDOM_SEEDS {
        require_success(
            &format!("randomized exchange graph seed {seed:#x}"),
            Command::new(env!("CARGO_BIN_EXE_ipu-randomized-e2e"))
                .env("IPU_RANDOM_SEED", format!("0x{seed:x}"))
                .status()
                .expect("launch randomized hardware test runner"),
        );
    }
}

fn device() -> MutexGuard<'static, ()> {
    DEVICE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn host_test(environment: &[(&str, &str)]) -> ExitStatus {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"));
    command.envs(environment.iter().copied());
    command.status().expect("launch host exchange test runner")
}

fn require_success(capability: &str, status: ExitStatus) {
    assert!(
        status.success(),
        "hardware capability {capability:?} failed with {status}"
    );
}
