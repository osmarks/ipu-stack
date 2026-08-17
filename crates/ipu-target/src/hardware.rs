//! Supported hardware targets and their resource limits.

use crate::cost::{HardwareCosts, IPU21_TARGET_COSTS};
use crate::memory::{
    IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES, IPU21_INTERLEAVED_ELEMENT_SIZE,
    IPU21_INTERLEAVED_REGION_BYTES, IPU21_PLANNED_DATA_BYTES, IPU21_STANDARD_FIXED_BYTES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardwareTarget {
    Ipu21,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HardwareMemoryConstraints {
    pub standard_fixed_bytes: u64,
    pub interleaved_bytes: u64,
    pub interleaved_element_bytes: u64,
    pub total_bytes: u64,
    pub default_standard_reservation_bytes: u64,
}

impl HardwareTarget {
    pub const fn costs(self) -> &'static HardwareCosts {
        match self {
            Self::Ipu21 => &IPU21_TARGET_COSTS,
        }
    }

    pub const fn memory_constraints(self) -> HardwareMemoryConstraints {
        match self {
            Self::Ipu21 => HardwareMemoryConstraints {
                standard_fixed_bytes: IPU21_STANDARD_FIXED_BYTES as u64,
                interleaved_bytes: IPU21_INTERLEAVED_REGION_BYTES as u64,
                interleaved_element_bytes: IPU21_INTERLEAVED_ELEMENT_SIZE as u64,
                total_bytes: IPU21_PLANNED_DATA_BYTES as u64,
                default_standard_reservation_bytes: IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES as u64,
            },
        }
    }
}
