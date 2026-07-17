use std::process::Command;

#[test]
#[ignore = "requires exclusive access to a physical C600"]
fn generated_compute_exchange_graph_runs_end_to_end() {
    let status = Command::new(env!("CARGO_BIN_EXE_ipu-hardware-e2e"))
        .status()
        .expect("launch single-threaded C600 test runner");
    assert!(status.success(), "C600 test runner failed with {status}");
}
