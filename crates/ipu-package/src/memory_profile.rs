use crate::{PackageError, memory_profile_capnp};
use capnp::{message, serialize};
use std::io::{Read, Write};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRegion {
    pub address: u32,
    pub size: u32,
    pub category: String,
    pub name: String,
    pub tensor: Option<usize>,
    pub live_from: usize,
    pub live_until: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileMemory {
    pub logical_tile: u16,
    pub physical_tile: u16,
    pub regions: Vec<MemoryRegion>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryProfile {
    pub memory_base: u32,
    pub memory_size: u32,
    pub tiles: Vec<TileMemory>,
}

impl MemoryProfile {
    pub fn write(&self, mut output: impl Write) -> Result<(), PackageError> {
        let mut message = message::Builder::new_default();
        let mut root = message.init_root::<memory_profile_capnp::memory_profile::Builder>();
        root.set_schema_version(1);
        root.set_memory_base(self.memory_base);
        root.set_memory_size(self.memory_size);
        let mut tiles = root.reborrow().init_tiles(self.tiles.len() as u32);
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            let mut output_tile = tiles.reborrow().get(tile_index as u32);
            output_tile.set_logical_tile(u32::from(tile.logical_tile));
            output_tile.set_physical_tile(u32::from(tile.physical_tile));
            let mut regions = output_tile
                .reborrow()
                .init_regions(tile.regions.len() as u32);
            for (region_index, region) in tile.regions.iter().enumerate() {
                let mut output_region = regions.reborrow().get(region_index as u32);
                output_region.set_address(region.address);
                output_region.set_size(region.size);
                output_region.set_category(&region.category);
                output_region.set_name(&region.name);
                output_region.set_has_tensor(region.tensor.is_some());
                output_region.set_tensor(region.tensor.map_or(0, |tensor| tensor as u64));
                output_region.set_live_from(region.live_from as u64);
                output_region.set_live_until(region.live_until as u64);
            }
        }
        serialize::write_message(&mut output, &message)?;
        Ok(())
    }

    pub fn read(mut input: impl Read) -> Result<Self, PackageError> {
        let message = serialize::read_message(&mut input, message::ReaderOptions::new())?;
        let root = message.get_root::<memory_profile_capnp::memory_profile::Reader>()?;
        if root.get_schema_version() != 1 {
            return Err(PackageError::Invalid(format!(
                "unsupported memory profile schema version {}",
                root.get_schema_version()
            )));
        }
        let tiles = root
            .get_tiles()?
            .iter()
            .map(|tile| {
                let logical_tile = u16::try_from(tile.get_logical_tile())
                    .map_err(|_| PackageError::Invalid("logical tile exceeds u16".into()))?;
                let physical_tile = u16::try_from(tile.get_physical_tile())
                    .map_err(|_| PackageError::Invalid("physical tile exceeds u16".into()))?;
                let regions = tile
                    .get_regions()?
                    .iter()
                    .map(|region| {
                        Ok(MemoryRegion {
                            address: region.get_address(),
                            size: region.get_size(),
                            category: region.get_category()?.to_str()?.into(),
                            name: region.get_name()?.to_str()?.into(),
                            tensor: region
                                .get_has_tensor()
                                .then(|| usize::try_from(region.get_tensor()))
                                .transpose()
                                .map_err(|_| {
                                    PackageError::Invalid("tensor identifier exceeds usize".into())
                                })?,
                            live_from: usize::try_from(region.get_live_from()).map_err(|_| {
                                PackageError::Invalid("live-from phase exceeds usize".into())
                            })?,
                            live_until: usize::try_from(region.get_live_until()).map_err(|_| {
                                PackageError::Invalid("live-until phase exceeds usize".into())
                            })?,
                        })
                    })
                    .collect::<Result<_, PackageError>>()?;
                Ok(TileMemory {
                    logical_tile,
                    physical_tile,
                    regions,
                })
            })
            .collect::<Result<_, PackageError>>()?;
        Ok(Self {
            memory_base: root.get_memory_base(),
            memory_size: root.get_memory_size(),
            tiles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_profile_round_trip() {
        let profile = MemoryProfile {
            memory_base: 0x4c000,
            memory_size: 624 * 1024,
            tiles: vec![TileMemory {
                logical_tile: 3,
                physical_tile: 67,
                regions: vec![MemoryRegion {
                    address: 0xa0000,
                    size: 4096,
                    category: "home".into(),
                    name: "left".into(),
                    tensor: Some(9),
                    live_from: 0,
                    live_until: usize::MAX,
                }],
            }],
        };
        let mut bytes = Vec::new();
        profile.write(&mut bytes).unwrap();
        assert_eq!(MemoryProfile::read(bytes.as_slice()).unwrap(), profile);
    }
}
