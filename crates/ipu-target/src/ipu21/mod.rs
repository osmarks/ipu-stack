//! IPU21 architectural facts. Runtime addresses, loader limits and compiler
//! preferences belong to their respective ABI and policy owners.
pub mod fabric;
pub mod instruction;
pub mod memory;
pub mod registers;

/// Worker contexts sharing one tile supervisor.
pub const WORKER_CONTEXTS: u32 = 6;
