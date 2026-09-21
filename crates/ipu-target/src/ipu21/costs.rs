//! Analytical prices for IPU21 and its resident worker runtime.

pub const COSTS: crate::TargetCosts = crate::TargetCosts {
    exchange_bytes_per_cycle: 4,
    local_copy_bytes_per_cycle: 8,
    // Finalized six-worker copy, including supervisor/worker rendezvous.
    local_copy_call_cycles: 288,
    exchange_phase_cycles: 600,
    kernel_launch_cycles: 11,
    send_control_cycles: 2,
    receive_control_cycles: 2,
    receive_pointer_cycles: 1,
};
