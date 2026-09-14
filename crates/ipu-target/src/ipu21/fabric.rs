//! Physical routing, pairing and tile inventory. The route-time equations
//! follow the recovered SDK IPU21 fabric model; schedule gaps and row encoding
//! are separate policies owned by ipu-exchange.
use std::collections::HashSet;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TopologyError {
    #[error("logical tile {0} is out of range")]
    Tile(u16),
    #[error("tile mapping must be nonempty and injective")]
    InvalidMapping,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    logical_to_physical: Vec<u16>,
}

impl Topology {
    pub fn new(logical_to_physical: Vec<u16>) -> Result<Self, TopologyError> {
        let mut physical = HashSet::new();
        if logical_to_physical.is_empty()
            || logical_to_physical
                .iter()
                .any(|tile| !physical.insert(*tile))
        {
            return Err(TopologyError::InvalidMapping);
        }
        Ok(Self {
            logical_to_physical,
        })
    }

    pub fn c600() -> Self {
        Self {
            logical_to_physical: (0..crate::c600::COMPUTE_TILES)
                .map(crate::c600::logical_to_physical)
                .collect(),
        }
    }

    pub fn tile_count(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn physical(&self, logical: u16) -> Result<u16, TopologyError> {
        self.logical_to_physical
            .get(usize::from(logical))
            .copied()
            .ok_or(TopologyError::Tile(logical))
    }

    /// Logical tile that shares this tile's double-width exchange resources.
    pub fn paired_logical(&self, logical: u16) -> Result<u16, TopologyError> {
        let paired_physical = self.physical(logical)? ^ 2;
        self.logical_to_physical
            .iter()
            .position(|physical| *physical == paired_physical)
            .map(|paired| u16::try_from(paired).expect("logical tile count fits u16"))
            .ok_or(TopologyError::Tile(logical))
    }

    /// Physical source selected by `INCOMING_MUXPAIR` for a 64-bit send.
    pub fn paired_source_mux(&self, sender_logical: u16) -> Result<u16, TopologyError> {
        Ok(self.physical(sender_logical)? ^ 2)
    }

    /// Whether this member of a double-width receiving pair owns the paired
    /// XPIC source-selection stream. This matches the SDK architecture
    /// helper `TPair_RxIsEarly`.
    pub fn paired_receiver_is_early(
        &self,
        receiver_logical: u16,
        sender_logical: u16,
    ) -> Result<bool, TopologyError> {
        let receiver = u32::from(self.physical(receiver_logical)?);
        let sender = u32::from(self.physical(sender_logical)?);
        let local = time_to_mux(sender, receiver);
        let borrowed = time_to_mux(sender, receiver ^ 2);
        Ok(local < borrowed)
    }
}

pub fn route_displacement(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_column = if source_raw > 7 {
        source_raw ^ 15
    } else {
        source_raw
    };
    let destination_column = if destination_raw > 7 {
        destination_raw ^ 15
    } else {
        destination_raw
    };
    let source_mux = source_column + ((source_raw >> 3) ^ (source & 1) as i32);
    let base = (destination_column - source_mux) * 6;
    let destination_lane = destination & 3;
    let destination_half = destination_raw >> 3;
    if destination_lane > 1 {
        base + if destination_half == (destination & 1) as i32 {
            2
        } else {
            4
        }
    } else {
        base + if destination_half == destination_lane as i32 {
            1
        } else {
            5
        }
    }
}

pub fn direction(source: u32, destination: u32) -> u32 {
    if route_displacement(source, destination) < 1 {
        2
    } else {
        1
    }
}

pub fn time_to_mux(source: u32, destination: u32) -> i32 {
    let source_raw = ((source >> 2) & 15) as i32;
    let destination_raw = ((destination >> 2) & 15) as i32;
    let source_low = ((source >> 2) & 7) as i32;
    let displacement = route_displacement(source, destination);
    let source_edge = if source_raw > 7 {
        (source_raw * 4) ^ 60
    } else {
        source_raw * 4
    };
    let destination_edge = if destination_raw > 7 {
        (destination_raw * 4) ^ 60
    } else {
        destination_raw * 4
    };
    let local = ((source >> 2) & 8) as i32 | ((source >> 3) & 3) as i32;
    let crossing = local - destination_raw + ((source_low >> 1) ^ 3);
    let same_region = (source ^ destination) & 0x20 == 0;
    let turn = if same_region {
        source_low + 1
    } else {
        16 - source_low
    };
    let group_delta = (((source >> 6) & 31) as i32 - ((destination >> 6) & 31) as i32) * 2;
    crossing + source_edge + turn - destination_edge + group_delta + displacement.abs() - 34
}

pub fn paired_time_to_mux(source: u32, destination: u32) -> i32 {
    time_to_mux(source, destination).max(time_to_mux(source, destination ^ 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn c600_mapping_is_a_permutation() {
        let topology = Topology::c600();
        let physical: HashSet<_> = (0..topology.tile_count() as u16)
            .map(|logical| topology.physical(logical).unwrap())
            .collect();
        assert_eq!(physical.len(), 1472);
        assert_eq!(topology.physical(46).unwrap(), 1409);
        assert!(physical.iter().all(|tile| *tile < 1472));
    }
}
