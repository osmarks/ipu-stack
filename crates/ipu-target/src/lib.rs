//! Device architecture and tile inventory, independent of compilation,
//! scheduling, package formats and the host driver.
pub mod c600;
pub mod ipu21;

/// Architecture selected for compilation. Board tile inventory is separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Ipu21,
}

impl Target {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ipu21 => "ipu21",
        }
    }

    pub const fn costs(self) -> &'static TargetCosts {
        match self {
            Self::Ipu21 => &ipu21::costs::COSTS,
        }
    }

    pub const fn tile_memory(self) -> std::ops::Range<u32> {
        match self {
            Self::Ipu21 => {
                ipu21::memory::TILE_MEMORY_BASE
                    ..ipu21::memory::TILE_MEMORY_BASE + ipu21::memory::TILE_MEMORY_SIZE
            }
        }
    }

    pub const fn standard_memory_element_bytes(self) -> u32 {
        match self {
            Self::Ipu21 => ipu21::memory::TILE_MEMORY_ELEMENT_SIZE,
        }
    }

    pub const fn interleaved_memory_element_bytes(self) -> u32 {
        match self {
            Self::Ipu21 => ipu21::memory::IPU21_INTERLEAVED_ELEMENT_SIZE,
        }
    }

    pub fn interleaved_data_bytes(self) -> u64 {
        match self {
            Self::Ipu21 => u64::from(ipu21::memory::IPU21_INTERLEAVED_REGION_BYTES),
        }
    }

    pub fn planned_data_bytes(self) -> u64 {
        match self {
            Self::Ipu21 => u64::from(ipu21::memory::IPU21_PLANNED_DATA_BYTES),
        }
    }

    pub fn support_reservation_bytes(self) -> u64 {
        match self {
            Self::Ipu21 => u64::from(ipu21::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
        }
    }

    /// Optimistic contiguous capacity, before alignment and bank placement.
    pub fn maximum_standard_allocation(self, reserved_standard_bytes: u64) -> u64 {
        match self {
            Self::Ipu21 => {
                // Standard buffers can borrow all of region 1 while interleaved
                // temporaries are dead; an unrelated class peak is not deducted.
                let lower = u64::from(ipu21::memory::IPU21_STANDARD_FIXED_BYTES)
                    .saturating_sub(reserved_standard_bytes);
                lower.max(self.interleaved_data_bytes())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetCosts {
    pub exchange_bytes_per_cycle: u64,
    pub local_copy_bytes_per_cycle: u64,
    pub local_copy_call_cycles: u64,
    pub exchange_phase_cycles: u64,
    pub kernel_launch_cycles: u64,
    pub send_control_cycles: u64,
    pub receive_control_cycles: u64,
    pub receive_pointer_cycles: u64,
}

impl std::str::FromStr for Target {
    type Err = &'static str;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "ipu21" => Ok(Self::Ipu21),
            _ => Err("unsupported IPU target"),
        }
    }
}
