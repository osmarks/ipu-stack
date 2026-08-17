//! IPU21 machine-level definitions and executable program generation.

pub mod cost;
pub mod emit;
pub mod exchange;
pub mod hardware;
pub mod instruction;
pub mod memory;
pub mod program;
pub mod topology;

pub use hardware::{HardwareMemoryConstraints, HardwareTarget};
