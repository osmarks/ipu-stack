//! Resident runtime placement and calling convention. These defaults are
//! shared with device/static_runtime.S; they are not target memory capacities.
include!(concat!(env!("OUT_DIR"), "/runtime_layout.rs"));
pub const RUNTIME_STATE_BYTES: u32 =
    WORKER_STACK_HEADROOM + super::WORKER_CONTEXTS * WORKER_SYNC_STRIDE;
pub const WORKER_BARRIER_SYMBOL: &str = "ipu_stack_static_worker_barrier";
pub const COMPLETE_SYMBOL: &str = "ipu_stack_static_complete";
pub const COMPLETED_SYMBOL: &str = "ipu_stack_static_completed";
pub const HOST_RUN_SYMBOL: &str = "ipu_stack_static_host_run";
pub const REPEAT_CALL_SYMBOL: &str = "ipu_stack_static_repeat_call";
pub const SAMPLE_CYCLE_SYMBOL: &str = "ipu_stack_static_sample_cycle";
pub const PATCH_REPEAT_TABLES_SYMBOL: &str = "static_patch_repeat_tables";
pub const PATCH_REPEAT_ARITHMETIC_SYMBOL: &str = "static_patch_repeat_arithmetic";
pub const PATCH_ROW_SYMBOL: &str = "ipu_stack_static_patch_row";
pub const RUNTIME_ENTRY_SYMBOL: &str = "ipu_stack_static_start";
pub const PROGRAM_ADDRESS_SYMBOL: &str = "ipu_stack_static_program";
pub const WORKER_SYNC_CONTEXT_SYMBOL: &str = "ipu_stack_static_worker_sync_context";
pub const WORKER_STACK_BASE_SYMBOL: &str = "ipu_stack_static_worker_stack_base";
pub const PRNG_SEED_SYMBOL: &str = "ipu_stack_static_prng_seed";
pub const HOST_STAGING_SYMBOL: &str = "ipu_stack_static_host_staging";
pub const COMPLETION_ADDRESS_SYMBOL: &str = "ipu_stack_static_completion";
