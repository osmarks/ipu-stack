//! Select parameter ownership rotations before expanding the low schedule.

use super::*;
use crate::storage::{StorageError, TensorStorage, storage_bytes};

impl MidProgram {
    pub(super) fn assign_parameter_tiles(&mut self) -> LoweringResult<()> {
        let parameter_origins = self
            .inputs
            .iter()
            .filter(|input| input.kind == GraphInputKind::Parameter)
            .map(|input| self.values[input.value.index() as usize].origin)
            .collect::<BTreeSet<_>>();
        let parameter_groups = self
            .values
            .iter()
            .filter(|value| parameter_origins.contains(&value.origin))
            .map(|value| value.storage_group)
            .collect::<BTreeSet<_>>();
        let mut loads = vec![0u64; usize::from(self.tile_count)];
        let mut offsets = BTreeMap::new();
        for value in &mut self.values {
            let layout = &value.tensor_type.format.layout;
            layout.validate_tile_count(self.tile_count)?;
            if !parameter_groups.contains(&value.storage_group) {
                continue;
            }
            let extents = layout.shard_extents(&value.tensor_type.shape)?;
            let bytes = extents
                .iter()
                .map(|(_, extents)| {
                    storage_bytes(TensorStorage {
                        format: &value.tensor_type.format,
                        extents,
                    })
                    .map(u64::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offset = *offsets
                .entry(value.storage_group)
                .or_insert_with(|| balanced_offset(&loads, &bytes));
            value.tile_offset = offset;
            for ((owner, _), bytes) in extents.iter().zip(bytes) {
                let tile = (usize::from(*owner) + usize::from(offset)) % loads.len();
                loads[tile] = loads[tile]
                    .checked_add(bytes)
                    .ok_or(StorageError::Overflow)?;
            }
        }
        Ok(())
    }
}

fn balanced_offset(loads: &[u64], bytes: &[u64]) -> u16 {
    let peak = loads.iter().copied().max().unwrap_or(0);
    // Each shard affects a different tile. The old peak covers unchanged tiles,
    // so candidate scoring needs neither a cloned load array nor a full rescan.
    (0..loads.len())
        .min_by_key(|&offset| {
            let peak = bytes
                .iter()
                .enumerate()
                .fold(peak, |peak, (logical, &bytes)| {
                    peak.max(loads[(logical + offset) % loads.len()].saturating_add(bytes))
                });
            (peak, offset)
        })
        .unwrap_or(0) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_match_full_load_array_scoring() {
        let mut random = fastrand::Rng::with_seed(0x0074_696c_6573);
        for _ in 0..1000 {
            let tiles = random.usize(1..=64);
            let loads = (0..tiles).map(|_| random.u64(0..4096)).collect::<Vec<_>>();
            let bytes = (0..random.usize(0..=tiles))
                .map(|_| random.u64(0..4096))
                .collect::<Vec<_>>();
            let expected = (0..tiles)
                .min_by_key(|&offset| {
                    let mut candidate = loads.clone();
                    for (logical, &bytes) in bytes.iter().enumerate() {
                        let tile = (logical + offset) % tiles;
                        candidate[tile] = candidate[tile].saturating_add(bytes);
                    }
                    (candidate.into_iter().max().unwrap(), offset)
                })
                .unwrap() as u16;
            assert_eq!(balanced_offset(&loads, &bytes), expected);
        }
    }
}
