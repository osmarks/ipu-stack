use std::process::{Command, ExitStatus};

const RANDOM_SEEDS: [u64; 3] = [
    0x4950_552d_5354_4143,
    0x6a09_e667_f3bc_c909,
    0xbb67_ae85_84ca_a73b,
];

#[test]
fn hardware_capabilities_run_end_to_end() {
    require_success(
        "initialized tile D2H",
        host_test(&[("IPU_HOST_TEST_D2H_ONLY", "1")]),
    );
    require_success("local H2D and D2H", host_test(&[]));
    require_success(
        "multi-epoch H2D, D2D, and D2H",
        host_test(&[
            ("IPU_HOST_TEST_EXCHANGE", "1"),
            ("IPU_HOST_TEST_BYTES", "2048"),
        ]),
    );
    require_success(
        "remote-tile H2D and D2H",
        host_test(&[
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_BYTES", "8192"),
        ]),
    );
    require_success(
        "remote-tile initialized D2H",
        host_test(&[
            ("IPU_HOST_TEST_REMOTE_D2H", "1"),
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_BYTES", "8192"),
        ]),
    );
    require_success(
        "distinct multi-source D2H",
        host_test(&[
            ("IPU_HOST_TEST_TILE", "1471"),
            ("IPU_HOST_TEST_SECOND_TILE", "274"),
            ("IPU_HOST_TEST_BYTES", "64"),
        ]),
    );
    require_success(
        "reduction, permutation, and multicast graph",
        Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
            .status()
            .expect("launch deterministic hardware test runner"),
    );
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
