use capnp::{message, serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use tracing::{info, trace};

pub mod application_capnp {
    include!(concat!(env!("OUT_DIR"), "/application_capnp.rs"));
}

pub mod profile_capnp {
    include!(concat!(env!("OUT_DIR"), "/profile_capnp.rs"));
}

fn capnp_reader_options() -> message::ReaderOptions {
    let mut options = message::ReaderOptions::new();
    options.traversal_limit_in_words(None);
    options
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileStepKind {
    Exchange,
    Compute,
    Synchronization,
    Idle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileStep {
    pub local_index: u32,
    pub phase: u32,
    pub epoch: u32,
    pub operation: String,
    pub kind: ProfileStepKind,
    pub kernel: String,
    pub metadata: Vec<ProfileMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileMetadata {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleSample {
    pub step: ProfileStep,
    pub start_cycle: u32,
    pub end_cycle: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileProfile {
    pub physical_tile: u32,
    pub samples: Vec<CycleSample>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileReport {
    pub clock_hz: u64,
    pub tiles: Vec<TileProfile>,
}

impl ProfileReport {
    pub fn write(&self, mut output: impl Write) -> Result<(), PackageError> {
        let mut message = message::Builder::new_default();
        let mut root = message.init_root::<profile_capnp::profile::Builder>();
        root.set_schema_version(3);
        root.set_clock_hz(self.clock_hz);
        let mut tiles = root.reborrow().init_tiles(self.tiles.len() as u32);
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            let mut output_tile = tiles.reborrow().get(tile_index as u32);
            output_tile.set_physical_tile(tile.physical_tile);
            let mut samples = output_tile
                .reborrow()
                .init_samples(tile.samples.len() as u32);
            for (sample_index, sample) in tile.samples.iter().enumerate() {
                let mut output_sample = samples.reborrow().get(sample_index as u32);
                output_sample.set_start_cycle(sample.start_cycle);
                output_sample.set_end_cycle(sample.end_cycle);
                let mut step = output_sample.reborrow().init_step();
                step.set_local_index(sample.step.local_index);
                step.set_phase(sample.step.phase);
                step.set_epoch(sample.step.epoch);
                step.set_operation(&sample.step.operation);
                step.set_kind(match sample.step.kind {
                    ProfileStepKind::Exchange => profile_capnp::StepKind::Exchange,
                    ProfileStepKind::Compute => profile_capnp::StepKind::Compute,
                    ProfileStepKind::Synchronization => profile_capnp::StepKind::Synchronization,
                    ProfileStepKind::Idle => profile_capnp::StepKind::Idle,
                });
                step.set_kernel(&sample.step.kernel);
                let mut metadata = step
                    .reborrow()
                    .init_metadata(sample.step.metadata.len() as u32);
                for (index, entry) in sample.step.metadata.iter().enumerate() {
                    let mut output_entry = metadata.reborrow().get(index as u32);
                    output_entry.set_name(&entry.name);
                    output_entry.set_value(&entry.value);
                }
            }
        }
        serialize::write_message(&mut output, &message)?;
        Ok(())
    }

    pub fn read(mut input: impl Read) -> Result<Self, PackageError> {
        let message = serialize::read_message(&mut input, capnp_reader_options())?;
        let root = message.get_root::<profile_capnp::profile::Reader>()?;
        if !matches!(root.get_schema_version(), 1..=3) {
            return Err(PackageError::Invalid(format!(
                "unsupported profile schema version {}",
                root.get_schema_version()
            )));
        }
        let tiles = root
            .get_tiles()?
            .iter()
            .map(|tile| {
                let samples = tile
                    .get_samples()?
                    .iter()
                    .map(|sample| {
                        let step = sample.get_step()?;
                        Ok(CycleSample {
                            step: ProfileStep {
                                local_index: step.get_local_index(),
                                phase: step.get_phase(),
                                epoch: step.get_epoch(),
                                operation: step.get_operation()?.to_str()?.into(),
                                kind: match step.get_kind()? {
                                    profile_capnp::StepKind::Exchange => ProfileStepKind::Exchange,
                                    profile_capnp::StepKind::Compute => ProfileStepKind::Compute,
                                    profile_capnp::StepKind::Synchronization => {
                                        ProfileStepKind::Synchronization
                                    }
                                    profile_capnp::StepKind::Idle => ProfileStepKind::Idle,
                                },
                                kernel: step.get_kernel()?.to_str()?.into(),
                                metadata: step
                                    .get_metadata()?
                                    .iter()
                                    .map(|entry| {
                                        Ok(ProfileMetadata {
                                            name: entry.get_name()?.to_str()?.into(),
                                            value: entry.get_value()?.to_str()?.into(),
                                        })
                                    })
                                    .collect::<Result<_, PackageError>>()?,
                            },
                            start_cycle: sample.get_start_cycle(),
                            end_cycle: sample.get_end_cycle(),
                        })
                    })
                    .collect::<Result<_, PackageError>>()?;
                Ok(TileProfile {
                    physical_tile: tile.get_physical_tile(),
                    samples,
                })
            })
            .collect::<Result<_, PackageError>>()?;
        Ok(Self {
            clock_hz: root.get_clock_hz(),
            tiles,
        })
    }
}

pub const SCHEMA_VERSION: u32 = 3;
pub const TARGET_IPU21: &str = "ipu21";
pub const TILE_MEMORY_BASE: u32 = 0x4c000;
pub const TILE_MEMORY_SIZE: u32 = 624 * 1024;
/// IPU21 `TMEM_ELEMSIZE`. Instruction fetch and data access contend at this
/// granularity even when placement policy is supplied by another crate.
pub const TILE_MEMORY_ELEMENT_SIZE: u32 = 0x4000;
/// Maximum supervisor instruction-fetch lookahead used when checking whether
/// executable and data ranges share a memory element.
pub const IPU21_SUPERVISOR_FETCH_LOOKAHEAD: u32 = 8 * 8;
/// End of IPU21 region 0, the only tile-memory region supporting instruction fetch.
pub const IPU21_EXECUTABLE_MEMORY_LIMIT: u32 = 0x80000;
/// First logical address of the commonly used interleaved operand window.
pub const IPU21_INTERLEAVED_MEMORY_BASE: u32 = TILE_MEMORY_BASE + 0x34000;
/// End of the commonly used interleaved operand window.
pub const IPU21_INTERLEAVED_MEMORY_LIMIT: u32 = TILE_MEMORY_BASE + 0x3c000;
/// End of architectural region 1, whose interleave factor is two on IPU21.
pub const IPU21_INTERLEAVED_REGION_LIMIT: u32 = TILE_MEMORY_BASE + TILE_MEMORY_SIZE;
/// Logical bytes covered by a pair of physical elements in interleaved region 1.
pub const IPU21_INTERLEAVED_ELEMENT_SIZE: u32 = 2 * TILE_MEMORY_ELEMENT_SIZE;
pub const SEGMENT_READ: u32 = 1;
pub const SEGMENT_WRITE: u32 = 2;
pub const SEGMENT_EXECUTE: u32 = 4;
pub const PROFILE_CYCLES_BINDING: &str = "profile.cycles";

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
pub struct Segment {
    pub address: u32,
    pub memory_size: u32,
    pub data: Vec<u8>,
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
    pub invocations: u32,
    /// Exclusive slice indices for each rolling-buffer input batch.
    pub input_batch_ends: Vec<u32>,
    /// Exclusive slice indices for each rolling-buffer output batch.
    pub output_batch_ends: Vec<u32>,
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
    /// Host-visible syncs after the initial application-startup rendezvous.
    pub external_syncs: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceConfigWrite {
    pub offset: u32,
    pub value: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileProfilePlan {
    pub physical_tile: u32,
    pub steps: Vec<ProfileStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Application {
    pub compiler_version: String,
    pub tiles: Vec<TileImage>,
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub weights: Vec<Binding>,
    pub host_exchange: HostExchange,
    pub entry_points: Vec<EntryPoint>,
    pub device_config_writes: Vec<DeviceConfigWrite>,
    pub profile_tiles: Vec<TileProfilePlan>,
}

impl Default for Application {
    fn default() -> Self {
        Self {
            compiler_version: env!("CARGO_PKG_VERSION").into(),
            tiles: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            weights: Vec::new(),
            host_exchange: HostExchange::default(),
            entry_points: Vec::new(),
            device_config_writes: Vec::new(),
            profile_tiles: Vec::new(),
        }
    }
}

impl Application {
    pub fn validate(&self) -> Result<(), PackageError> {
        if self.tiles.is_empty() {
            return Err(PackageError::Invalid("application has no tiles".into()));
        }
        let mut config_offsets = std::collections::HashSet::new();
        if self
            .device_config_writes
            .iter()
            .any(|write| write.offset & 3 != 0 || !config_offsets.insert(write.offset))
        {
            return Err(PackageError::Invalid(
                "unaligned or duplicate device configuration write".into(),
            ));
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
                    || segment.data.len() > segment.memory_size as usize
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
            if let Some(pair) = ranges.windows(2).find(|pair| pair[0].1 > pair[1].0) {
                return Err(PackageError::Invalid(format!(
                    "overlapping segments on tile {}: 0x{:x}..0x{:x} and 0x{:x}..0x{:x}",
                    tile.physical_tile, pair[0].0, pair[0].1, pair[1].0, pair[1].1
                )));
            }
        }
        let tile_ids: std::collections::HashSet<_> =
            self.tiles.iter().map(|tile| tile.physical_tile).collect();
        let mut profile_tile_ids = std::collections::HashSet::new();
        for tile in &self.profile_tiles {
            if !tile_ids.contains(&tile.physical_tile)
                || !profile_tile_ids.insert(tile.physical_tile)
                || tile
                    .steps
                    .iter()
                    .enumerate()
                    .any(|(index, step)| step.local_index != index as u32)
            {
                return Err(PackageError::Invalid("invalid tile profile plan".into()));
            }
        }
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
            return Err(PackageError::Invalid(
                "invalid host startup protocol".into(),
            ));
        }
        for call in &self.host_exchange.calls {
            if call.invocations == 0 {
                return Err(PackageError::Invalid(format!(
                    "host call {} has no invocations",
                    call.name
                )));
            }
            validate_host_batch_ends(call, "input", &call.input_batch_ends, call.inputs.len())?;
            validate_host_batch_ends(call, "output", &call.output_batch_ends, call.outputs.len())?;
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
        info!(tiles = self.tiles.len(), "writing application package");
        let mut message = message::Builder::new_default();
        let mut root = message.init_root::<application_capnp::application::Builder>();
        root.set_schema_version(SCHEMA_VERSION);
        root.set_compiler_version(&self.compiler_version);
        root.set_target(TARGET_IPU21);
        root.set_tile_memory_base(TILE_MEMORY_BASE);
        root.set_tile_memory_size(TILE_MEMORY_SIZE);

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
            item.set_external_syncs(entry.external_syncs);
        }
        let mut config_writes = root
            .reborrow()
            .init_device_config_writes(self.device_config_writes.len() as u32);
        for (index, write) in self.device_config_writes.iter().enumerate() {
            let mut item = config_writes.reborrow().get(index as u32);
            item.set_offset(write.offset);
            item.set_value(write.value);
        }
        write_profile_tiles(
            root.reborrow()
                .init_profile_tiles(self.profile_tiles.len() as u32),
            &self.profile_tiles,
        );
        serialize::write_message(&mut output, &message)?;
        info!("application package written");
        Ok(())
    }

    pub fn read(mut input: impl Read) -> Result<Self, PackageError> {
        info!("reading application package");
        let reader = serialize::read_message(&mut input, capnp_reader_options())?;
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
                    external_syncs: item.get_external_syncs(),
                })
            })
            .collect::<Result<_, PackageError>>()?;
        app.device_config_writes = root
            .get_device_config_writes()?
            .iter()
            .map(|item| DeviceConfigWrite {
                offset: item.get_offset(),
                value: item.get_value(),
            })
            .collect();
        app.profile_tiles = read_profile_tiles(root.get_profile_tiles()?)?;
        app.validate()?;
        info!(
            tiles = app.tiles.len(),
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
            image[destination..destination + segment.data.len()].copy_from_slice(&segment.data);
        }
        Ok(image)
    }

    pub fn profile_report(
        &self,
        output: &[u8],
        clock_hz: u64,
    ) -> Result<ProfileReport, PackageError> {
        let mut binding_base = 0u64;
        let mut profile_binding = None;
        for binding in &self.outputs {
            if binding.name == PROFILE_CYCLES_BINDING {
                profile_binding = Some(binding);
                break;
            }
            binding_base = binding_base
                .checked_add(binding_size(binding))
                .ok_or_else(|| PackageError::Invalid("output binding size overflow".into()))?;
        }
        let binding = profile_binding.ok_or_else(|| {
            PackageError::Invalid("application has no cycle profile binding".into())
        })?;
        let slices = binding
            .slices
            .iter()
            .map(|slice| (slice.tile, slice))
            .collect::<HashMap<_, _>>();
        let mut tiles = Vec::with_capacity(self.profile_tiles.len());
        for plan in &self.profile_tiles {
            let slice = slices.get(&plan.physical_tile).ok_or_else(|| {
                PackageError::Invalid(format!(
                    "cycle profile has no slice for tile {}",
                    plan.physical_tile
                ))
            })?;
            let expected = u64::try_from(plan.steps.len() + 1)
                .map_err(|_| PackageError::Invalid("profile sample count overflow".into()))?
                .checked_mul(4)
                .ok_or_else(|| PackageError::Invalid("profile sample size overflow".into()))?;
            if slice.size != expected {
                return Err(PackageError::Invalid(format!(
                    "cycle profile slice for tile {} has size {}, expected {expected}",
                    plan.physical_tile, slice.size
                )));
            }
            let start = usize::try_from(binding_base + slice.file_offset)
                .map_err(|_| PackageError::Invalid("profile output offset overflow".into()))?;
            let end = start
                .checked_add(usize::try_from(slice.size).map_err(|_| {
                    PackageError::Invalid("profile output size exceeds usize".into())
                })?)
                .ok_or_else(|| PackageError::Invalid("profile output range overflow".into()))?;
            let bytes = output.get(start..end).ok_or_else(|| {
                PackageError::Invalid("profile binding exceeds runtime output".into())
            })?;
            let cycles = bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>();
            tiles.push(TileProfile {
                physical_tile: plan.physical_tile,
                samples: plan
                    .steps
                    .iter()
                    .cloned()
                    .zip(cycles.windows(2))
                    .map(|(step, bounds)| CycleSample {
                        step,
                        start_cycle: bounds[0],
                        end_cycle: bounds[1],
                    })
                    .collect(),
            });
        }
        Ok(ProfileReport { clock_hz, tiles })
    }
}

fn binding_size(binding: &Binding) -> u64 {
    binding
        .slices
        .iter()
        .map(|slice| slice.file_offset.saturating_add(slice.size))
        .max()
        .unwrap_or(0)
}
fn validate_host_batch_ends(
    call: &HostCall,
    direction: &str,
    ends: &[u32],
    slice_count: usize,
) -> Result<(), PackageError> {
    let mut previous = 0usize;
    for &end in ends {
        let end = usize::try_from(end)
            .map_err(|_| PackageError::Invalid("host batch index exceeds usize".into()))?;
        if end <= previous || end > slice_count {
            return Err(PackageError::Invalid(format!(
                "host call {} has invalid {direction} batch boundary {end}",
                call.name
            )));
        }
        previous = end;
    }
    if previous != slice_count {
        return Err(PackageError::Invalid(format!(
            "host call {} {direction} batches cover {previous} of {slice_count} slices",
            call.name
        )));
    }
    Ok(())
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
            out.set_data(&segment.data);
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
                    .map(|segment| {
                        Ok(Segment {
                            address: segment.get_address(),
                            memory_size: segment.get_memory_size(),
                            data: segment.get_data()?.to_vec(),
                            flags: segment.get_flags(),
                        })
                    })
                    .collect::<Result<_, capnp::Error>>()?,
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

fn write_profile_tiles(
    mut output: capnp::struct_list::Builder<'_, application_capnp::tile_profile_plan::Owned>,
    tiles: &[TileProfilePlan],
) {
    for (tile_index, tile) in tiles.iter().enumerate() {
        let mut output_tile = output.reborrow().get(tile_index as u32);
        output_tile.set_physical_tile(tile.physical_tile);
        let mut steps = output_tile.reborrow().init_steps(tile.steps.len() as u32);
        for (step_index, step) in tile.steps.iter().enumerate() {
            let mut output_step = steps.reborrow().get(step_index as u32);
            output_step.set_local_index(step.local_index);
            output_step.set_phase(step.phase);
            output_step.set_epoch(step.epoch);
            output_step.set_operation(&step.operation);
            output_step.set_kind(match step.kind {
                ProfileStepKind::Exchange => application_capnp::ProfileStepKind::Exchange,
                ProfileStepKind::Compute => application_capnp::ProfileStepKind::Compute,
                ProfileStepKind::Synchronization => {
                    application_capnp::ProfileStepKind::Synchronization
                }
                ProfileStepKind::Idle => application_capnp::ProfileStepKind::Idle,
            });
            output_step.set_kernel(&step.kernel);
            let mut metadata = output_step
                .reborrow()
                .init_metadata(step.metadata.len() as u32);
            for (metadata_index, entry) in step.metadata.iter().enumerate() {
                let mut output_entry = metadata.reborrow().get(metadata_index as u32);
                output_entry.set_name(&entry.name);
                output_entry.set_value(&entry.value);
            }
        }
    }
}

fn read_profile_tiles(
    input: capnp::struct_list::Reader<'_, application_capnp::tile_profile_plan::Owned>,
) -> Result<Vec<TileProfilePlan>, PackageError> {
    input
        .iter()
        .map(|tile| {
            Ok(TileProfilePlan {
                physical_tile: tile.get_physical_tile(),
                steps: tile
                    .get_steps()?
                    .iter()
                    .map(|step| {
                        Ok(ProfileStep {
                            local_index: step.get_local_index(),
                            phase: step.get_phase(),
                            epoch: step.get_epoch(),
                            operation: step.get_operation()?.to_str()?.into(),
                            kind: match step.get_kind()? {
                                application_capnp::ProfileStepKind::Exchange => {
                                    ProfileStepKind::Exchange
                                }
                                application_capnp::ProfileStepKind::Compute => {
                                    ProfileStepKind::Compute
                                }
                                application_capnp::ProfileStepKind::Synchronization => {
                                    ProfileStepKind::Synchronization
                                }
                                application_capnp::ProfileStepKind::Idle => ProfileStepKind::Idle,
                            },
                            kernel: step.get_kernel()?.to_str()?.into(),
                            metadata: step
                                .get_metadata()?
                                .iter()
                                .map(|entry| {
                                    Ok(ProfileMetadata {
                                        name: entry.get_name()?.to_str()?.into(),
                                        value: entry.get_value()?.to_str()?.into(),
                                    })
                                })
                                .collect::<Result<_, PackageError>>()?,
                        })
                    })
                    .collect::<Result<_, PackageError>>()?,
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
        item.set_invocations(call.invocations);
        let mut input_batch_ends = item
            .reborrow()
            .init_input_batch_ends(call.input_batch_ends.len() as u32);
        for (index, end) in call.input_batch_ends.iter().enumerate() {
            input_batch_ends.set(index as u32, *end);
        }
        let mut output_batch_ends = item
            .reborrow()
            .init_output_batch_ends(call.output_batch_ends.len() as u32);
        for (index, end) in call.output_batch_ends.iter().enumerate() {
            output_batch_ends.set(index as u32, *end);
        }
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
                    invocations: call.get_invocations(),
                    inputs: read_host_slices(call.get_inputs()?),
                    outputs: read_host_slices(call.get_outputs()?),
                    input_batch_ends: call.get_input_batch_ends()?.iter().collect(),
                    output_batch_ends: call.get_output_batch_ends()?.iter().collect(),
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
        app.tiles.push(TileImage {
            physical_tile: 0,
            entry_point: TILE_MEMORY_BASE,
            command_address: TILE_MEMORY_BASE + 0x100,
            diagnostic_address: TILE_MEMORY_BASE + 0x200,
            segments: vec![Segment {
                address: TILE_MEMORY_BASE,
                memory_size: 8,
                data: vec![1, 2, 3, 4],
                flags: SEGMENT_READ | SEGMENT_EXECUTE,
            }],
        });
        app
    }

    #[test]
    fn round_trip_and_reconstruct() {
        let mut app = sample();
        app.profile_tiles.push(TileProfilePlan {
            physical_tile: 0,
            steps: vec![ProfileStep {
                local_index: 0,
                phase: 7,
                epoch: 0,
                operation: "operation.0".into(),
                kind: ProfileStepKind::Compute,
                kernel: "kernel".into(),
                metadata: vec![ProfileMetadata {
                    name: "reason".into(),
                    value: "OperatorKernel".into(),
                }],
            }],
        });
        app.device_config_writes.push(DeviceConfigWrite {
            offset: 0x4018,
            value: 0xc000_000d,
        });
        app.host_exchange = HostExchange {
            startup_mark: 1,
            command_page: 0,
            command_offset: 0,
            pages: vec![
                HostPage {
                    index: 0,
                    size: 4096,
                },
                HostPage {
                    index: 1,
                    size: 8192,
                },
            ],
            attach_order: vec![0, 1],
            calls: vec![HostCall {
                name: "batched".into(),
                command: 2,
                phases: 4,
                inputs: vec![
                    HostSlice {
                        page: 1,
                        page_offset: 0,
                        file_offset: 0,
                        size: 16,
                    },
                    HostSlice {
                        page: 1,
                        page_offset: 4096,
                        file_offset: 16,
                        size: 16,
                    },
                ],
                outputs: vec![HostSlice {
                    page: 1,
                    page_offset: 0,
                    file_offset: 0,
                    size: 32,
                }],
                invocations: 1,
                input_batch_ends: vec![2],
                output_batch_ends: vec![1],
            }],
        };
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
    fn randomized_profile_reports_use_adjacent_start_samples() {
        let mut random = fastrand::Rng::with_seed(0x7072_6f66);
        for case in 0..64 {
            let tile_count = random.usize(1..=8);
            let prefix_bytes = random.usize(0..=8) * 4;
            let mut app = Application::default();
            app.outputs.push(Binding {
                name: "prefix".into(),
                dtype: "u8".into(),
                shape: vec![prefix_bytes as u32],
                slices: (prefix_bytes != 0)
                    .then_some(RegionSlice {
                        tile: 0,
                        tile_address: TILE_MEMORY_BASE,
                        file_offset: 0,
                        size: prefix_bytes as u64,
                    })
                    .into_iter()
                    .collect(),
            });
            let mut output = vec![0xa5; prefix_bytes];
            let mut slices = Vec::new();
            let mut profile_offset = 0u64;
            let mut expected = Vec::new();
            for tile in 0..tile_count {
                let step_count = random.usize(1..=16);
                let mut cycle = random.u32(0..=u32::MAX / 2);
                let mut bounds = vec![cycle];
                for _ in 0..step_count {
                    cycle = cycle.wrapping_add(random.u32(1..=10_000));
                    bounds.push(cycle);
                }
                output.extend(bounds.iter().flat_map(|cycle| cycle.to_le_bytes()));
                slices.push(RegionSlice {
                    tile: tile as u32,
                    tile_address: TILE_MEMORY_BASE + 0x100,
                    file_offset: profile_offset,
                    size: ((step_count + 1) * 4) as u64,
                });
                profile_offset += ((step_count + 1) * 4) as u64;
                let steps = (0..step_count)
                    .map(|index| ProfileStep {
                        local_index: index as u32,
                        phase: index as u32,
                        epoch: 0,
                        operation: format!("operation.{index}"),
                        kind: if random.bool() {
                            ProfileStepKind::Compute
                        } else {
                            ProfileStepKind::Exchange
                        },
                        kernel: format!("kernel.{index}"),
                        metadata: Vec::new(),
                    })
                    .collect::<Vec<_>>();
                expected.push((tile as u32, steps.clone(), bounds));
                app.profile_tiles.push(TileProfilePlan {
                    physical_tile: tile as u32,
                    steps,
                });
            }
            app.outputs.push(Binding {
                name: PROFILE_CYCLES_BINDING.into(),
                dtype: "u32".into(),
                shape: vec![(profile_offset / 4) as u32],
                slices,
            });

            let report = app.profile_report(&output, 1_500_000_000).unwrap();
            assert_eq!(report.tiles.len(), tile_count, "random case {case}");
            for (tile_index, (tile, (physical_tile, steps, bounds))) in
                report.tiles.iter().zip(expected).enumerate()
            {
                assert_eq!(tile_index as u32, physical_tile, "random case {case}");
                assert_eq!(tile.physical_tile, physical_tile);
                for (index, sample) in tile.samples.iter().enumerate() {
                    assert_eq!(sample.step, steps[index], "random case {case}");
                    assert_eq!(sample.start_cycle, bounds[index], "random case {case}");
                    assert_eq!(sample.end_cycle, bounds[index + 1], "random case {case}");
                }
            }
        }
    }

    #[test]
    fn rejects_overlapping_segments() {
        let mut app = sample();
        let duplicate = app.tiles[0].segments[0].clone();
        app.tiles[0].segments.push(duplicate);
        assert!(app.validate().is_err());
    }

    #[test]
    fn rejects_ambiguous_device_configuration() {
        let mut app = sample();
        app.device_config_writes = vec![
            DeviceConfigWrite {
                offset: 4,
                value: 1,
            },
            DeviceConfigWrite {
                offset: 4,
                value: 2,
            },
        ];
        assert!(app.validate().is_err());
    }

    #[test]
    fn rejects_implicit_host_batches() {
        let call = HostCall {
            name: "explicit-batches".into(),
            command: 0,
            phases: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            invocations: 1,
            input_batch_ends: Vec::new(),
            output_batch_ends: Vec::new(),
        };
        assert!(validate_host_batch_ends(&call, "input", &[], 1).is_err());
        assert!(validate_host_batch_ends(&call, "output", &[], 1).is_err());
        assert!(validate_host_batch_ends(&call, "input", &[], 0).is_ok());
    }

    #[test]
    fn profile_round_trip() {
        let profile = ProfileReport {
            clock_hz: 1_500_000_000,
            tiles: vec![TileProfile {
                physical_tile: 17,
                samples: [
                    ProfileStepKind::Exchange,
                    ProfileStepKind::Compute,
                    ProfileStepKind::Synchronization,
                    ProfileStepKind::Idle,
                ]
                .into_iter()
                .enumerate()
                .map(|(index, kind)| CycleSample {
                    step: ProfileStep {
                        local_index: index as u32,
                        phase: 5,
                        epoch: 8,
                        operation: "accumulate".into(),
                        kind,
                        kernel: "gemm_f32_accumulate".into(),
                        metadata: vec![ProfileMetadata {
                            name: "innerBlock".into(),
                            value: "8".into(),
                        }],
                    },
                    start_cycle: (u32::MAX - 10).wrapping_add(index as u32),
                    end_cycle: 7 + index as u32,
                })
                .collect(),
            }],
        };
        let mut encoded = Vec::new();
        profile.write(&mut encoded).unwrap();
        assert_eq!(ProfileReport::read(encoded.as_slice()).unwrap(), profile);
    }
}
