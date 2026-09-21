//! Device architecture and tile inventory, independent of compilation,
//! scheduling, package formats and the host driver.
pub mod c600;
pub mod ipu21;

/// Architecture selected for compilation. Board tile inventory is separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Ipu21,
}

impl Target {
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
