//! IPU21 architecture and the shared runtime/loader contracts used on this target.
pub mod fabric;
pub mod instruction;
pub mod loader_abi;
pub mod memory;
pub mod registers;
pub mod runtime_layout;

/// Worker contexts sharing one tile supervisor.
pub const WORKER_CONTEXTS: u32 = 6;
