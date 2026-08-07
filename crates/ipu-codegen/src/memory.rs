/// Runtime completion word followed by the supervisor and worker stack state.
pub const RUNTIME_STATE_BASE: u32 =
    ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
pub const WORKER_STACK_HEADROOM: u32 = 0xe0;
pub const WORKER_SYNC_STRIDE: u32 = 0x100;
pub const WORKER_CONTEXTS: u32 = 6;
pub const RUNTIME_STATE_BYTES: u32 = WORKER_STACK_HEADROOM + WORKER_CONTEXTS * WORKER_SYNC_STRIDE;

/// First byte after the permanently reserved runtime state.
pub const IPU21_DATA_BASE: u32 = RUNTIME_STATE_BASE + RUNTIME_STATE_BYTES;
