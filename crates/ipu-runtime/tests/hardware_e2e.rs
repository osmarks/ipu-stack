use std::process::Command;

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_compute_exchange_graph_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
        .status()
        .expect("launch single-threaded C600 test runner");
    assert!(status.success(), "C600 test runner failed with {status}");
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_host_exchange_graph_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .status()
        .expect("launch C600 host exchange test runner");
    assert!(
        status.success(),
        "C600 host test runner failed with {status}"
    );
}

#[test]
fn generated_d2h_from_initialized_tile_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .env("IPU_HOST_TEST_D2H_ONLY", "1")
        .status()
        .expect("launch initialized-tile C600 D2H test runner");
    assert!(
        status.success(),
        "C600 initialized-tile D2H test runner failed with {status}"
    );
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_multi_epoch_host_exchange_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .env("IPU_HOST_TEST_EXCHANGE", "1")
        .env("IPU_HOST_TEST_BYTES", "2048")
        .status()
        .expect("launch multi-epoch C600 host exchange test runner");
    assert!(
        status.success(),
        "C600 multi-epoch host exchange test runner failed with {status}"
    );
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_remote_tile_h2d_and_d2h_run_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .env("IPU_HOST_TEST_TILE", "1471")
        .env("IPU_HOST_TEST_BYTES", "8192")
        .status()
        .expect("launch remote-tile C600 H2D/D2H test runner");
    assert!(
        status.success(),
        "C600 remote H2D/D2H test runner failed with {status}"
    );
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_remote_tile_d2h_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .env("IPU_HOST_TEST_REMOTE_D2H", "1")
        .env("IPU_HOST_TEST_TILE", "1471")
        .env("IPU_HOST_TEST_BYTES", "8192")
        .status()
        .expect("launch remote-tile C600 D2H test runner");
    assert!(
        status.success(),
        "C600 remote D2H test runner failed with {status}"
    );
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_distinct_multi_source_d2h_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-host-e2e"))
        .env("IPU_HOST_TEST_TILE", "1471")
        .env("IPU_HOST_TEST_SECOND_TILE", "274")
        .env("IPU_HOST_TEST_BYTES", "64")
        .status()
        .expect("launch distinct multi-source C600 D2H test runner");
    assert!(
        status.success(),
        "C600 distinct multi-source D2H test runner failed with {status}"
    );
}

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn randomized_exchange_graphs_run_end_to_end() {
    for seed in [
        0x4950_552d_5354_4143u64,
        0x6a09_e667_f3bc_c909,
        0xbb67_ae85_84ca_a73b,
    ] {
        let status = Command::new(env!("CARGO_BIN_EXE_ipu-randomized-e2e"))
            .env("IPU_RANDOM_SEED", format!("0x{seed:x}"))
            .status()
            .expect("launch randomized C600 exchange test runner");
        assert!(
            status.success(),
            "C600 randomized exchange test failed for seed {seed:#x} with {status}"
        );
    }
}
