use capnp::{message, serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use tracing::{debug, info, trace};

pub mod application_capnp {
    include!(concat!(env!("OUT_DIR"), "/application_capnp.rs"));
}

pub const SCHEMA_VERSION: u32 = 1;
pub const TARGET_IPU21: &str = "ipu21";
pub const TILE_MEMORY_BASE: u32 = 0x4c000;
pub const TILE_MEMORY_SIZE: u32 = 624 * 1024;
pub const SEGMENT_READ: u32 = 1;
pub const SEGMENT_WRITE: u32 = 2;
pub const SEGMENT_EXECUTE: u32 = 4;

#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    #[error("Cap'n Proto error: {0}")]
    Capnp(#[from] capnp::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid UTF-8 text: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("unknown schema enum value: {0}")]
    SchemaEnum(#[from] capnp::NotInSchema),
    #[error("invalid package: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blob {
    pub digest: [u8; 32],
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    pub address: u32,
    pub memory_size: u32,
    pub blob: usize,
    pub blob_offset: u64,
    pub file_size: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileImage {
    pub physical_tile: u32,
    pub entry_point: u32,
    pub command_address: u32,
    pub diagnostic_address: u32,
    pub segments: Vec<Segment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionSlice {
    pub tile: u32,
    pub tile_address: u32,
    pub file_offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub slices: Vec<RegionSlice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPage {
    pub index: u32,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSlice {
    pub page: u32,
    pub page_offset: u64,
    pub file_offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCall {
    pub name: String,
    pub command: u32,
    pub phases: u32,
    pub inputs: Vec<HostSlice>,
    pub outputs: Vec<HostSlice>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct HostExchange {
    pub startup_mark: u32,
    pub command_page: u32,
    pub command_offset: u64,
    pub pages: Vec<HostPage>,
    pub attach_order: Vec<u32>,
    pub calls: Vec<HostCall>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryPoint {
    pub name: String,
    pub command: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Application {
    pub compiler_version: String,
    pub blobs: Vec<Blob>,
    pub tiles: Vec<TileImage>,
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub weights: Vec<Binding>,
    pub host_exchange: HostExchange,
    pub entry_points: Vec<EntryPoint>,
}

impl Default for Application {
    fn default() -> Self {
        Self {
            compiler_version: env!("CARGO_PKG_VERSION").into(),
            blobs: Vec::new(),
            tiles: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: Vec::new(),
            host_exchange: HostExchange::default(),
            entry_points: Vec::new(),
        }
    }
}

impl Application {
    pub fn add_blob(&mut self, bytes: Vec<u8>) -> usize {
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if let Some(index) = self.blobs.iter().position(|blob| blob.digest == digest) {
            return index;
        }
        self.blobs.push(Blob { digest, bytes });
        self.blobs.len() - 1
    }

    pub fn validate(&self) -> Result<(), PackageError> {
        if self.tiles.is_empty() {
            return Err(PackageError::Invalid("application has no tiles".into()));
        }
        let mut seen = HashMap::new();
        for tile in &self.tiles {
            if seen.insert(tile.physical_tile, ()).is_some() {
                return Err(PackageError::Invalid(format!(
                    "duplicate physical tile {}",
                    tile.physical_tile
                )));
            }
            for segment in &tile.segments {
                let end = segment
                    .address
                    .checked_add(segment.memory_size)
                    .ok_or_else(|| PackageError::Invalid("segment address overflow".into()))?;
                if segment.address < TILE_MEMORY_BASE
                    || end > TILE_MEMORY_BASE + TILE_MEMORY_SIZE
                    || segment.file_size > segment.memory_size
                    || segment.blob >= self.blobs.len()
                    || segment.blob_offset + u64::from(segment.file_size)
                        > self.blobs[segment.blob].bytes.len() as u64
                {
                    return Err(PackageError::Invalid(format!(
                        "invalid segment on tile {}",
                        tile.physical_tile
                    )));
                }
            }
            let mut ranges: Vec<_> = tile
                .segments
                .iter()
                .filter(|segment| segment.memory_size != 0)
                .map(|segment| (segment.address, segment.address + segment.memory_size))
                .collect();
            ranges.sort_unstable();
            if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
                return Err(PackageError::Invalid(format!(
                    "overlapping segments on tile {}",
                    tile.physical_tile
                )));
            }
        }
        let tile_ids: std::collections::HashSet<_> =
            self.tiles.iter().map(|tile| tile.physical_tile).collect();
        let mut binding_names = std::collections::HashSet::new();
        for binding in self.inputs.iter().chain(&self.outputs).chain(&self.weights) {
            if binding.name.is_empty() || !binding_names.insert(binding.name.as_str()) {
                return Err(PackageError::Invalid(format!(
                    "empty or duplicate binding {}",
                    binding.name
                )));
            }
            for slice in &binding.slices {
                let end = slice
                    .tile_address
                    .checked_add(u32::try_from(slice.size).map_err(|_| {
                        PackageError::Invalid(format!(
                            "binding {} slice is too large",
                            binding.name
                        ))
                    })?)
                    .ok_or_else(|| PackageError::Invalid("binding address overflow".into()))?;
                if !tile_ids.contains(&slice.tile)
                    || slice.tile_address < TILE_MEMORY_BASE
                    || end > TILE_MEMORY_BASE + TILE_MEMORY_SIZE
                {
                    return Err(PackageError::Invalid(format!(
                        "binding {} references invalid tile memory",
                        binding.name
                    )));
                }
            }
        }
        let pages: HashMap<_, _> = self
            .host_exchange
            .pages
            .iter()
            .map(|page| (page.index, page.size))
            .collect();
        if pages.len() != self.host_exchange.pages.len()
            || self
                .host_exchange
                .attach_order
                .iter()
                .any(|index| !pages.contains_key(index))
        {
            return Err(PackageError::Invalid("invalid host page table".into()));
        }
        if !self.host_exchange.calls.is_empty()
            && pages
                .get(&self.host_exchange.command_page)
                .is_none_or(|size| self.host_exchange.command_offset.checked_add(4) > Some(*size))
        {
            return Err(PackageError::Invalid("invalid host command word".into()));
        }
        for call in &self.host_exchange.calls {
            for slice in call.inputs.iter().chain(&call.outputs) {
                let Some(page_size) = pages.get(&slice.page) else {
                    return Err(PackageError::Invalid(format!(
                        "host call {} references missing page",
                        call.name
                    )));
                };
                if slice.page_offset.checked_add(slice.size) > Some(*page_size) {
                    return Err(PackageError::Invalid(format!(
                        "host call {} exceeds page bounds",
                        call.name
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn write(&self, mut output: impl Write) -> Result<(), PackageError> {
        self.validate()?;
        info!(
            tiles = self.tiles.len(),
            blobs = self.blobs.len(),
            "writing application package"
        );
        let mut message = message::Builder::new_default();
        let mut root = message.init_root::<application_capnp::application::Builder>();
        root.set_schema_version(SCHEMA_VERSION);
        root.set_compiler_version(&self.compiler_version);
        root.set_target(TARGET_IPU21);
        root.set_tile_memory_base(TILE_MEMORY_BASE);
        root.set_tile_memory_size(TILE_MEMORY_SIZE);

        let mut blobs = root.reborrow().init_blobs(self.blobs.len() as u32);
        for (index, blob) in self.blobs.iter().enumerate() {
            let mut item = blobs.reborrow().get(index as u32);
            item.set_sha256(&blob.digest);
            item.set_uncompressed_size(blob.bytes.len() as u64);
            let compressed = zstd::bulk::compress(&blob.bytes, 3)?;
            debug!(
                blob = index,
                raw_bytes = blob.bytes.len(),
                compressed_bytes = compressed.len(),
                "encoded package blob"
            );
            if compressed.len() < blob.bytes.len() {
                item.set_codec(application_capnp::BlobCodec::Zstd);
                item.set_data(&compressed);
            } else {
                item.set_codec(application_capnp::BlobCodec::Raw);
                item.set_data(&blob.bytes);
            }
        }
        write_tiles(
            root.reborrow().init_tiles(self.tiles.len() as u32),
            &self.tiles,
        );
        write_bindings(
            root.reborrow().init_inputs(self.inputs.len() as u32),
            &self.inputs,
        );
        write_bindings(
            root.reborrow().init_outputs(self.outputs.len() as u32),
            &self.outputs,
        );
        write_bindings(
            root.reborrow().init_weights(self.weights.len() as u32),
            &self.weights,
        );
        write_host_exchange(root.reborrow().init_host_exchange(), &self.host_exchange);
        let mut entries = root
            .reborrow()
            .init_entry_points(self.entry_points.len() as u32);
        for (index, entry) in self.entry_points.iter().enumerate() {
            let mut item = entries.reborrow().get(index as u32);
            item.set_name(&entry.name);
            item.set_command(entry.command);
        }
        let digest = self.build_digest();
        root.set_build_digest(&digest);
        serialize::write_message(&mut output, &message)?;
        info!(digest = %hex_digest(&digest), "application package written");
        Ok(())
    }

    pub fn read(mut input: impl Read) -> Result<Self, PackageError> {
        info!("reading application package");
        let reader = serialize::read_message(&mut input, message::ReaderOptions::new())?;
        let root = reader.get_root::<application_capnp::application::Reader>()?;
        if root.get_schema_version() != SCHEMA_VERSION
            || root.get_target()?.to_str()? != TARGET_IPU21
            || root.get_tile_memory_base() != TILE_MEMORY_BASE
            || root.get_tile_memory_size() != TILE_MEMORY_SIZE
        {
            return Err(PackageError::Invalid("incompatible package header".into()));
        }
        let mut app = Application {
            compiler_version: root.get_compiler_version()?.to_str()?.into(),
            ..Application::default()
        };
        for item in root.get_blobs()?.iter() {
            let stored = item.get_data()?;
            let bytes = match item.get_codec()? {
                application_capnp::BlobCodec::Raw => stored.to_vec(),
                application_capnp::BlobCodec::Zstd => {
                    zstd::bulk::decompress(stored, item.get_uncompressed_size() as usize)?
                }
            };
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            if item.get_sha256()? != digest {
                return Err(PackageError::Invalid("blob digest mismatch".into()));
            }
            app.blobs.push(Blob { digest, bytes });
        }
        app.tiles = read_tiles(root.get_tiles()?)?;
        app.inputs = read_bindings(root.get_inputs()?)?;
        app.outputs = read_bindings(root.get_outputs()?)?;
        app.weights = read_bindings(root.get_weights()?)?;
        app.host_exchange = read_host_exchange(root.get_host_exchange()?)?;
        app.entry_points = root
            .get_entry_points()?
            .iter()
            .map(|item| {
                Ok(EntryPoint {
                    name: item.get_name()?.to_str()?.into(),
                    command: item.get_command(),
                })
            })
            .collect::<Result<_, PackageError>>()?;
        app.validate()?;
        if root.get_build_digest()? != app.build_digest() {
            return Err(PackageError::Invalid("build digest mismatch".into()));
        }
        info!(
            tiles = app.tiles.len(),
            blobs = app.blobs.len(),
            compiler = %app.compiler_version,
            "application package read"
        );
        Ok(app)
    }

    pub fn tile_image(&self, physical_tile: u32) -> Result<Vec<u8>, PackageError> {
        let tile = self
            .tiles
            .iter()
            .find(|tile| tile.physical_tile == physical_tile)
            .ok_or_else(|| PackageError::Invalid(format!("unknown tile {physical_tile}")))?;
        let load_base = tile
            .segments
            .iter()
            .map(|segment| segment.address)
            .min()
            .ok_or_else(|| PackageError::Invalid("tile has no loadable segments".into()))?;
        let image_size = tile
            .segments
            .iter()
            .map(|segment| (segment.address - load_base + segment.memory_size) as usize)
            .max()
            .unwrap_or(0);
        let mut image = vec![0; image_size];
        trace!(
            physical_tile,
            load_base = format_args!("0x{load_base:x}"),
            image_bytes = image_size,
            "reconstructing tile image"
        );
        for segment in &tile.segments {
            let destination = (segment.address - load_base) as usize;
            let source = segment.blob_offset as usize;
            let size = segment.file_size as usize;
            image[destination..destination + size]
                .copy_from_slice(&self.blobs[segment.blob].bytes[source..source + size]);
        }
        Ok(image)
    }

    fn build_digest(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(SCHEMA_VERSION.to_le_bytes());
        hash_string(&mut hash, TARGET_IPU21);
        hash.update(TILE_MEMORY_BASE.to_le_bytes());
        hash.update(TILE_MEMORY_SIZE.to_le_bytes());
        hash_string(&mut hash, &self.compiler_version);
        hash_len(&mut hash, self.blobs.len());
        for blob in &self.blobs {
            hash.update(blob.digest);
        }
        hash_len(&mut hash, self.tiles.len());
        for tile in &self.tiles {
            hash.update(tile.physical_tile.to_le_bytes());
            hash.update(tile.entry_point.to_le_bytes());
            hash.update(tile.command_address.to_le_bytes());
            hash.update(tile.diagnostic_address.to_le_bytes());
            hash_len(&mut hash, tile.segments.len());
            for segment in &tile.segments {
                hash.update(segment.address.to_le_bytes());
                hash.update(segment.memory_size.to_le_bytes());
                hash.update((segment.blob as u64).to_le_bytes());
                hash.update(segment.blob_offset.to_le_bytes());
                hash.update(segment.file_size.to_le_bytes());
                hash.update(segment.flags.to_le_bytes());
            }
        }
        for bindings in [&self.inputs, &self.outputs, &self.weights] {
            hash.update((bindings.len() as u64).to_le_bytes());
            for binding in bindings {
                hash_string(&mut hash, &binding.name);
                hash_string(&mut hash, &binding.dtype);
                hash_len(&mut hash, binding.shape.len());
                for dimension in &binding.shape {
                    hash.update(dimension.to_le_bytes());
                }
                hash_len(&mut hash, binding.slices.len());
                for slice in &binding.slices {
                    hash.update(slice.tile.to_le_bytes());
                    hash.update(slice.tile_address.to_le_bytes());
                    hash.update(slice.file_offset.to_le_bytes());
                    hash.update(slice.size.to_le_bytes());
                }
            }
        }
        let host = &self.host_exchange;
        hash.update(host.startup_mark.to_le_bytes());
        hash.update(host.command_page.to_le_bytes());
        hash.update(host.command_offset.to_le_bytes());
        hash_len(&mut hash, host.pages.len());
        for page in &host.pages {
            hash.update(page.index.to_le_bytes());
            hash.update(page.size.to_le_bytes());
        }
        hash_len(&mut hash, host.attach_order.len());
        for page in &host.attach_order {
            hash.update(page.to_le_bytes());
        }
        hash_len(&mut hash, host.calls.len());
        for call in &host.calls {
            hash_string(&mut hash, &call.name);
            hash.update(call.command.to_le_bytes());
            hash.update(call.phases.to_le_bytes());
            for slices in [&call.inputs, &call.outputs] {
                hash_len(&mut hash, slices.len());
                for slice in slices {
                    hash.update(slice.page.to_le_bytes());
                    hash.update(slice.page_offset.to_le_bytes());
                    hash.update(slice.file_offset.to_le_bytes());
                    hash.update(slice.size.to_le_bytes());
                }
            }
        }
        hash_len(&mut hash, self.entry_points.len());
        for entry in &self.entry_points {
            hash_string(&mut hash, &entry.name);
            hash.update(entry.command.to_le_bytes());
        }
        hash.finalize().into()
    }
}

fn hash_string(hash: &mut Sha256, value: &str) {
    hash_len(hash, value.len());
    hash.update(value.as_bytes());
}

fn hash_len(hash: &mut Sha256, length: usize) {
    hash.update((length as u64).to_le_bytes());
}

fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

fn write_tiles(
    mut output: capnp::struct_list::Builder<'_, application_capnp::tile_image::Owned>,
    tiles: &[TileImage],
) {
    for (index, tile) in tiles.iter().enumerate() {
        let mut item = output.reborrow().get(index as u32);
        item.set_physical_tile(tile.physical_tile);
        item.set_entry_point(tile.entry_point);
        item.set_command_address(tile.command_address);
        item.set_diagnostic_address(tile.diagnostic_address);
        let mut segments = item.reborrow().init_segments(tile.segments.len() as u32);
        for (segment_index, segment) in tile.segments.iter().enumerate() {
            let mut out = segments.reborrow().get(segment_index as u32);
            out.set_address(segment.address);
            out.set_memory_size(segment.memory_size);
            out.set_blob_index(segment.blob as u32);
            out.set_blob_offset(segment.blob_offset);
            out.set_file_size(segment.file_size);
            out.set_flags(segment.flags);
        }
    }
}

fn read_tiles(
    input: capnp::struct_list::Reader<'_, application_capnp::tile_image::Owned>,
) -> Result<Vec<TileImage>, PackageError> {
    input
        .iter()
        .map(|item| {
            Ok(TileImage {
                physical_tile: item.get_physical_tile(),
                entry_point: item.get_entry_point(),
                command_address: item.get_command_address(),
                diagnostic_address: item.get_diagnostic_address(),
                segments: item
                    .get_segments()?
                    .iter()
                    .map(|segment| Segment {
                        address: segment.get_address(),
                        memory_size: segment.get_memory_size(),
                        blob: segment.get_blob_index() as usize,
                        blob_offset: segment.get_blob_offset(),
                        file_size: segment.get_file_size(),
                        flags: segment.get_flags(),
                    })
                    .collect(),
            })
        })
        .collect()
}

fn write_bindings(
    mut output: capnp::struct_list::Builder<'_, application_capnp::binding::Owned>,
    bindings: &[Binding],
) {
    for (index, binding) in bindings.iter().enumerate() {
        let mut item = output.reborrow().get(index as u32);
        item.set_name(&binding.name);
        item.set_dtype(&binding.dtype);
        let mut shape = item.reborrow().init_shape(binding.shape.len() as u32);
        for (axis, value) in binding.shape.iter().enumerate() {
            shape.set(axis as u32, *value);
        }
        let mut slices = item.reborrow().init_slices(binding.slices.len() as u32);
        for (slice_index, slice) in binding.slices.iter().enumerate() {
            let mut out = slices.reborrow().get(slice_index as u32);
            out.set_tile(slice.tile);
            out.set_tile_address(slice.tile_address);
            out.set_file_offset(slice.file_offset);
            out.set_size(slice.size);
        }
    }
}

fn read_bindings(
    input: capnp::struct_list::Reader<'_, application_capnp::binding::Owned>,
) -> Result<Vec<Binding>, PackageError> {
    input
        .iter()
        .map(|item| {
            Ok(Binding {
                name: item.get_name()?.to_str()?.into(),
                dtype: item.get_dtype()?.to_str()?.into(),
                shape: item.get_shape()?.iter().collect(),
                slices: item
                    .get_slices()?
                    .iter()
                    .map(|slice| RegionSlice {
                        tile: slice.get_tile(),
                        tile_address: slice.get_tile_address(),
                        file_offset: slice.get_file_offset(),
                        size: slice.get_size(),
                    })
                    .collect(),
            })
        })
        .collect()
}

fn write_host_exchange(
    mut output: application_capnp::host_exchange::Builder<'_>,
    host: &HostExchange,
) {
    output.set_startup_mark(host.startup_mark);
    output.set_command_page(host.command_page);
    output.set_command_offset(host.command_offset);
    let mut pages = output.reborrow().init_pages(host.pages.len() as u32);
    for (index, page) in host.pages.iter().enumerate() {
        let mut item = pages.reborrow().get(index as u32);
        item.set_index(page.index);
        item.set_size(page.size);
    }
    let mut order = output
        .reborrow()
        .init_attach_order(host.attach_order.len() as u32);
    for (index, page) in host.attach_order.iter().enumerate() {
        order.set(index as u32, *page);
    }
    let mut calls = output.reborrow().init_calls(host.calls.len() as u32);
    for (index, call) in host.calls.iter().enumerate() {
        let mut item = calls.reborrow().get(index as u32);
        item.set_name(&call.name);
        item.set_command(call.command);
        item.set_phases(call.phases);
        write_host_slices(
            item.reborrow().init_inputs(call.inputs.len() as u32),
            &call.inputs,
        );
        write_host_slices(
            item.reborrow().init_outputs(call.outputs.len() as u32),
            &call.outputs,
        );
    }
}

fn write_host_slices(
    mut output: capnp::struct_list::Builder<'_, application_capnp::host_slice::Owned>,
    slices: &[HostSlice],
) {
    for (index, slice) in slices.iter().enumerate() {
        let mut item = output.reborrow().get(index as u32);
        item.set_page(slice.page);
        item.set_page_offset(slice.page_offset);
        item.set_file_offset(slice.file_offset);
        item.set_size(slice.size);
    }
}

fn read_host_exchange(
    input: application_capnp::host_exchange::Reader<'_>,
) -> Result<HostExchange, PackageError> {
    Ok(HostExchange {
        startup_mark: input.get_startup_mark(),
        command_page: input.get_command_page(),
        command_offset: input.get_command_offset(),
        pages: input
            .get_pages()?
            .iter()
            .map(|page| HostPage {
                index: page.get_index(),
                size: page.get_size(),
            })
            .collect(),
        attach_order: input.get_attach_order()?.iter().collect(),
        calls: input
            .get_calls()?
            .iter()
            .map(|call| {
                Ok(HostCall {
                    name: call.get_name()?.to_str()?.into(),
                    command: call.get_command(),
                    phases: call.get_phases(),
                    inputs: read_host_slices(call.get_inputs()?),
                    outputs: read_host_slices(call.get_outputs()?),
                })
            })
            .collect::<Result<_, PackageError>>()?,
    })
}

fn read_host_slices(
    input: capnp::struct_list::Reader<'_, application_capnp::host_slice::Owned>,
) -> Vec<HostSlice> {
    input
        .iter()
        .map(|slice| HostSlice {
            page: slice.get_page(),
            page_offset: slice.get_page_offset(),
            file_offset: slice.get_file_offset(),
            size: slice.get_size(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Application {
        let mut app = Application::default();
        let blob = app.add_blob(vec![1, 2, 3, 4]);
        assert_eq!(blob, app.add_blob(vec![1, 2, 3, 4]));
        app.tiles.push(TileImage {
            physical_tile: 0,
            entry_point: TILE_MEMORY_BASE,
            command_address: TILE_MEMORY_BASE + 0x100,
            diagnostic_address: TILE_MEMORY_BASE + 0x200,
            segments: vec![Segment {
                address: TILE_MEMORY_BASE,
                memory_size: 8,
                blob,
                blob_offset: 0,
                file_size: 4,
                flags: SEGMENT_READ | SEGMENT_EXECUTE,
            }],
        });
        app
    }

    #[test]
    fn round_trip_and_reconstruct() {
        let app = sample();
        let mut encoded = Vec::new();
        app.write(&mut encoded).unwrap();
        let decoded = Application::read(encoded.as_slice()).unwrap();
        assert_eq!(decoded, app);
        assert_eq!(
            &decoded.tile_image(0).unwrap()[..8],
            &[1, 2, 3, 4, 0, 0, 0, 0]
        );
    }

    #[test]
    fn rejects_overlapping_segments() {
        let mut app = sample();
        let duplicate = app.tiles[0].segments[0].clone();
        app.tiles[0].segments.push(duplicate);
        assert!(app.validate().is_err());
    }
}
