use crate::graph::ComputeGraph;
use crate::{
    COMPLETE_SYMBOL, COMPLETION_ADDRESS_SYMBOL, CodegenOptions, HostProgram, PRNG_SEED_SYMBOL,
    PROGRAM_ADDRESS_SYMBOL, RUNTIME_ENTRY_SYMBOL, TileProgram, WORKER_STACK_BASE_SYMBOL,
    WORKER_SYNC_CONTEXT_SYMBOL, emit,
};
use ipu_driver::{APPLICATION_LOAD_BASE, TILES_PER_BATCH};
use ipu_elf::{ElfError, LinkOptions, LinkedImage, Toolchain, link};
use ipu_exchange::{ExchangeError, Topology, encode_br_m, encode_setzi_m};
use ipu_package::{
    Application, Binding, EntryPoint, PackageError, RegionSlice, SEGMENT_EXECUTE, SEGMENT_READ,
    SEGMENT_WRITE, Segment, TILE_MEMORY_BASE, TileImage,
};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::num::TryFromIntError;
use std::path::PathBuf;

const ENTRY_BYTES: u32 = 8;
const SUPPORT_START: u32 = APPLICATION_LOAD_BASE + ENTRY_BYTES;
const COMPLETION_ADDRESS: u32 =
    ipu_exchange::EXCHANGE_WINDOW_BASE + ipu_exchange::EXCHANGE_WINDOW_BYTES;
const WORKER_STACK_HEADROOM: u32 = 0xe0;
const WORKER_SYNC_STRIDE: u32 = 0x100;
const WORKER_CONTEXTS: u32 = 6;
const RUNTIME_STATE_BYTES: u32 = WORKER_STACK_HEADROOM + WORKER_CONTEXTS * WORKER_SYNC_STRIDE;

#[derive(Debug, thiserror::Error)]
pub enum PackageBuildError {
    #[error("code generation failed: {0}")]
    Codegen(#[from] crate::CodegenError),
    #[error("ELF processing failed: {0}")]
    Elf(#[from] ElfError),
    #[error("exchange encoding failed: {0}")]
    Exchange(#[from] ExchangeError),
    #[error("package construction failed: {0}")]
    Package(#[from] PackageError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer conversion failed: {0}")]
    Integer(#[from] TryFromIntError),
    #[error("invalid package build: {0}")]
    Invalid(String),
}

pub type PackageBuildResult<T> = std::result::Result<T, PackageBuildError>;

#[derive(Clone, Debug)]
pub struct PackageConfig {
    pub toolchain: Toolchain,
    pub runtime_source: PathBuf,
    pub build_directory: PathBuf,
    pub tile_count: u32,
}

/// Compiles and packages a compute graph into a directly loadable IPU21
/// application.
pub fn build_package(
    graph: &ComputeGraph,
    config: &PackageConfig,
) -> PackageBuildResult<Application> {
    validate_tile_count(config.tile_count)?;
    let artifact = config.toolchain.compile(
        &config.runtime_source,
        &config.build_directory,
        "static_runtime",
        &[],
    )?;
    let object = fs::read(&artifact.object)?;
    build_package_from_object(graph, config, &object)
}

fn build_package_from_object(
    graph: &ComputeGraph,
    config: &PackageConfig,
    runtime_object: &[u8],
) -> PackageBuildResult<Application> {
    let layout = link_runtime(runtime_object, runtime_symbols(0, 0)?)?;
    let code_address = align_up(linked_end(&layout)?, 4)?;
    let generated = emit(
        &lower_graph(graph),
        &BTreeMap::from([(COMPLETE_SYMBOL.into(), symbol(&layout, COMPLETE_SYMBOL)?)]),
        &HostProgram::default(),
        &CodegenOptions {
            code_address,
            ..CodegenOptions::default()
        },
    )?;
    let code_end = code_address
        .checked_add(u32::try_from(generated.bytes.len())?)
        .ok_or_else(|| invalid("generated code address overflow"))?;
    if code_end > ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT {
        return Err(invalid("generated program does not fit executable memory"));
    }

    let mut application = Application::default();
    for physical_tile in 0..config.tile_count {
        application.tiles.push(build_tile(
            physical_tile,
            runtime_object,
            code_address,
            &generated.bytes,
        )?);
    }
    application.outputs.push(Binding {
        name: "completion".into(),
        dtype: "u32".into(),
        shape: vec![1],
        slices: vec![RegionSlice {
            tile: 0,
            tile_address: COMPLETION_ADDRESS,
            file_offset: 0,
            size: 4,
        }],
    });
    application.entry_points.push(EntryPoint {
        name: "run".into(),
        command: 0,
        external_syncs: 0,
    });
    application.validate()?;
    Ok(application)
}

fn validate_tile_count(tile_count: u32) -> PackageBuildResult<()> {
    let maximum = Topology::c600().tile_count() as u32;
    if tile_count == 0 || !tile_count.is_multiple_of(TILES_PER_BATCH as u32) || tile_count > maximum
    {
        return Err(invalid(format!(
            "tile count must be a nonzero multiple of {TILES_PER_BATCH} and at most {maximum}"
        )));
    }
    Ok(())
}

fn build_tile(
    physical_tile: u32,
    runtime_object: &[u8],
    code_address: u32,
    generated: &[u8],
) -> PackageBuildResult<TileImage> {
    let linked = link_runtime(
        runtime_object,
        runtime_symbols(physical_tile, code_address)?,
    )?;
    let mut entry = Vec::with_capacity(ENTRY_BYTES as usize);
    entry.extend_from_slice(&encode_setzi_m(0, linked.entry)?.to_le_bytes());
    entry.extend_from_slice(&encode_br_m(0)?.to_le_bytes());
    let mut segments = vec![Segment {
        address: APPLICATION_LOAD_BASE,
        memory_size: ENTRY_BYTES,
        data: entry,
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }];
    segments.extend(linked.segments.iter().map(|segment| Segment {
        address: segment.address,
        memory_size: segment.size as u32,
        data: linked.bytes[segment.offset..segment.offset + segment.size].to_vec(),
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    }));
    segments.push(Segment {
        address: code_address,
        memory_size: generated.len() as u32,
        data: generated.to_vec(),
        flags: SEGMENT_READ | SEGMENT_EXECUTE,
    });
    segments.push(Segment {
        address: COMPLETION_ADDRESS,
        memory_size: RUNTIME_STATE_BYTES,
        data: vec![0; 4],
        flags: SEGMENT_READ | SEGMENT_WRITE,
    });
    Ok(TileImage {
        physical_tile,
        entry_point: APPLICATION_LOAD_BASE,
        command_address: 0,
        diagnostic_address: COMPLETION_ADDRESS,
        segments,
    })
}

fn link_runtime(object: &[u8], externals: HashMap<String, u32>) -> PackageBuildResult<LinkedImage> {
    Ok(link(
        &[object.to_vec()],
        &LinkOptions {
            image_base: TILE_MEMORY_BASE,
            regions: vec![(SUPPORT_START, ipu_package::IPU21_EXECUTABLE_MEMORY_LIMIT)],
            entry_symbol: RUNTIME_ENTRY_SYMBOL.into(),
            retained_symbols: vec![COMPLETE_SYMBOL.into()],
            externals,
        },
    )?)
}

fn runtime_symbols(
    physical_tile: u32,
    program_address: u32,
) -> PackageBuildResult<HashMap<String, u32>> {
    let sync_context = physical_tile
        .checked_mul(8)
        .ok_or_else(|| invalid("tile index overflow"))?;
    let prng_seed = physical_tile
        .checked_add(1)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid("PRNG seed overflow"))?;
    Ok(HashMap::from([
        (WORKER_SYNC_CONTEXT_SYMBOL.into(), sync_context),
        (
            WORKER_STACK_BASE_SYMBOL.into(),
            COMPLETION_ADDRESS + WORKER_STACK_HEADROOM,
        ),
        (PRNG_SEED_SYMBOL.into(), prng_seed),
        (PROGRAM_ADDRESS_SYMBOL.into(), program_address),
        (COMPLETION_ADDRESS_SYMBOL.into(), COMPLETION_ADDRESS),
    ]))
}

fn symbol(linked: &LinkedImage, name: &str) -> PackageBuildResult<u32> {
    linked
        .symbols
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("linked runtime has no {name} symbol")))
}

fn linked_end(linked: &LinkedImage) -> PackageBuildResult<u32> {
    linked
        .segments
        .iter()
        .map(|segment| segment.address.checked_add(segment.size as u32))
        .collect::<Option<Vec<_>>>()
        .and_then(|ends| ends.into_iter().max())
        .ok_or_else(|| invalid("linked runtime has no valid segments"))
}

fn align_up(value: u32, alignment: u32) -> PackageBuildResult<u32> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| invalid("address alignment overflow"))
}

fn invalid(message: impl Into<String>) -> PackageBuildError {
    PackageBuildError::Invalid(message.into())
}

fn lower_graph(_graph: &ComputeGraph) -> TileProgram {
    TileProgram::default()
}
