//! Empirical and architectural cycle costs for supported targets.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareCosts {
    pub exchange_bytes_per_cycle: u64,
    pub standard_load_bytes_per_cycle: u64,
    pub interleaved_load_bytes_per_cycle: u64,
    pub local_copy_bytes_per_cycle: u64,
    pub reduction_output_bytes_per_cycle: u64,
    pub local_copy_call_cycles: u64,
    pub exchange_phase_cycles: u64,
    pub kernel_launch_cycles: u64,
    pub logical_fragment_cycles: u64,
    pub amp_call_cycles: u64,
    pub amp_column_group_width: u64,
    pub amp_interleaved_column_group_cycles: u64,
    pub amp_standard_column_group_cycles: u64,
    pub indexed_f16_transform_cycles_per_element: u64,
    pub amp_left_pack_cycles_per_element: u64,
    pub contiguous_panel_pack_cycles_per_element: u64,
    pub block_major_pack_startup_cycles: u64,
    pub block_major_pack_cycles_per_element: u64,
}

/// IPU21 architectural costs and measurements used by analytical planning.
pub const IPU21_TARGET_COSTS: HardwareCosts = HardwareCosts {
    // Target::getExchangeBytesPerCycle.
    exchange_bytes_per_cycle: 4,
    // Target::getMemcpyBytesPerCycle. Interleaved reads use both memory
    // elements, while an ordinary read or local copy uses one data path.
    standard_load_bytes_per_cycle: 8,
    interleaved_load_bytes_per_cycle: 16,
    local_copy_bytes_per_cycle: 8,
    // Reduction-add reads two partials and writes one. Current profiles
    // sustain roughly one output byte per cycle after the three interleaved
    // streams and worker imbalance are included.
    reduction_output_bytes_per_cycle: 1,
    // Finalized six-worker local copy, including both rendezvous.
    local_copy_call_cycles: 288,
    // Target::getGlobalSyncCycles.
    exchange_phase_cycles: 600,
    // popops::internal::basicOpSupervisorOverhead(false).
    kernel_launch_cycles: 11,
    // Endpoint and receive-pointer cutovers in fragmented logical exchange.
    logical_fragment_cycles: 160,
    // Generated AMP GEMM kernel: one 16-column group and one 64-element K
    // block, with the remaining group cost dominated by weight delivery.
    amp_call_cycles: 294,
    amp_column_group_width: 16,
    amp_interleaved_column_group_cycles: 940,
    amp_standard_column_group_cycles: 1_063,
    // Generated F16 layout-transform measurements.
    indexed_f16_transform_cycles_per_element: 10,
    amp_left_pack_cycles_per_element: 4,
    contiguous_panel_pack_cycles_per_element: 3,
    block_major_pack_startup_cycles: 4_096,
    block_major_pack_cycles_per_element: 4,
};
