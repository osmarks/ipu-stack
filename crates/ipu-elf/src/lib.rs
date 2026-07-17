use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget, SectionKind, SymbolKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tracing::{debug, info};

pub const R_COLOSSUS_NONE: u32 = 0;
pub const R_COLOSSUS_8: u32 = 1;
pub const R_COLOSSUS_16: u32 = 2;
pub const R_COLOSSUS_20: u32 = 3;
pub const R_COLOSSUS_32: u32 = 4;
pub const R_COLOSSUS_64: u32 = 5;
pub const R_COLOSSUS_RELATIVE_16_S2: u32 = 6;
pub const R_COLOSSUS_18_S2: u32 = 7;
pub const R_COLOSSUS_19_S2: u32 = 8;
pub const R_COLOSSUS_RUN: u32 = 9;
pub const R_COLOSSUS_16_S3: u32 = 14;
pub const R_COLOSSUS_16_S4: u32 = 15;
pub const R_COLOSSUS_16_S5: u32 = 16;
pub const R_COLOSSUS_21: u32 = 17;

#[derive(Debug, thiserror::Error)]
pub enum ElfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("object error: {0}")]
    Object(#[from] object::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tool failed: {0}")]
    Tool(String),
    #[error("link error: {0}")]
    Link(String),
}

#[derive(Clone, Debug)]
pub struct Toolchain {
    pub popc: PathBuf,
    pub pop_objdump: PathBuf,
    pub target: String,
}

impl Toolchain {
    pub fn from_sdk(sdk: impl AsRef<Path>) -> Self {
        let bin = sdk.as_ref().join("bin");
        Self {
            popc: bin.join("popc"),
            pop_objdump: bin.join("pop-objdump"),
            target: "ipu21".into(),
        }
    }

    pub fn compile(
        &self,
        source: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        name: &str,
        flags: &[String],
    ) -> Result<KernelArtifact, ElfError> {
        info!(
            source = %source.as_ref().display(),
            name,
            target = %self.target,
            "compiling kernel"
        );
        fs::create_dir_all(output_dir.as_ref())?;
        let gp = output_dir.as_ref().join(format!("{name}.gp"));
        let object = output_dir.as_ref().join(format!("{name}.o"));
        let metadata = output_dir.as_ref().join(format!("{name}.json"));
        let mut command = Command::new(&self.popc);
        command
            .arg("--target")
            .arg(&self.target)
            .arg("-O2")
            .args(flags)
            .arg(source.as_ref())
            .arg("-o")
            .arg(&gp);
        run(&mut command, "popc")?;

        let object_file = fs::File::create(&object)?;
        let mut extract = Command::new(&self.pop_objdump);
        extract
            .arg("extract")
            .arg(&self.target)
            .arg(&gp)
            .stdout(Stdio::from(object_file));
        run(&mut extract, "pop-objdump extract")?;

        let metadata_file = fs::File::create(&metadata)?;
        let mut dump = Command::new(&self.pop_objdump);
        dump.arg("metadata")
            .arg(&self.target)
            .arg(&gp)
            .stdout(Stdio::from(metadata_file));
        run(&mut dump, "pop-objdump metadata")?;
        let artifact = KernelArtifact {
            gp,
            object,
            metadata,
        };
        artifact.inspect()?;
        info!(
            object = %artifact.object.display(),
            metadata = %artifact.metadata.display(),
            "kernel artifact created"
        );
        Ok(artifact)
    }
}

fn run(command: &mut Command, name: &str) -> Result<(), ElfError> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(ElfError::Tool(format!("{name} exited with {status}")))
    }
}

#[derive(Clone, Debug)]
pub struct KernelArtifact {
    pub gp: PathBuf,
    pub object: PathBuf,
    pub metadata: PathBuf,
}

impl KernelArtifact {
    pub fn inspect(&self) -> Result<ObjectSummary, ElfError> {
        inspect_object(&fs::read(&self.object)?)
    }

    pub fn digest(&self) -> Result<[u8; 32], ElfError> {
        Ok(Sha256::digest(fs::read(&self.object)?).into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectSummary {
    pub architecture: String,
    pub sections: Vec<String>,
    pub defined_symbols: Vec<String>,
    pub undefined_symbols: Vec<String>,
    pub relocation_types: Vec<u32>,
}

pub fn inspect_object(bytes: &[u8]) -> Result<ObjectSummary, ElfError> {
    if bytes.len() < 20 || &bytes[..4] != b"\x7fELF" || bytes[4] != 1 || bytes[5] != 1 {
        return Err(ElfError::Link(
            "kernel object is not little-endian ELF32".into(),
        ));
    }
    let machine = u16::from_le_bytes([bytes[18], bytes[19]]);
    if machine != 0xf8 {
        return Err(ElfError::Link(format!(
            "ELF machine 0x{machine:x} is not Colossus"
        )));
    }
    let file = object::File::parse(bytes)?;
    let mut defined = Vec::new();
    let mut undefined = Vec::new();
    for symbol in file.symbols() {
        let name = symbol.name().unwrap_or_default();
        if name.is_empty() || symbol.kind() == SymbolKind::File {
            continue;
        }
        if symbol.is_undefined() {
            undefined.push(name.into());
        } else if symbol.is_definition() {
            defined.push(name.into());
        }
    }
    defined.sort();
    undefined.sort();
    let mut relocations = Vec::new();
    for section in file.sections() {
        for (_, relocation) in section.relocations() {
            if let object::RelocationFlags::Elf { r_type } = relocation.flags() {
                relocations.push(r_type);
            }
        }
    }
    relocations.sort_unstable();
    relocations.dedup();
    Ok(ObjectSummary {
        architecture: "Colossus".into(),
        sections: file
            .sections()
            .filter_map(|section| section.name().ok().map(str::to_owned))
            .collect(),
        defined_symbols: defined,
        undefined_symbols: undefined,
        relocation_types: relocations,
    })
}

#[derive(Clone, Debug)]
pub struct LinkOptions {
    pub image_base: u32,
    pub entry_symbol: String,
    /// Symbols reached through runtime dispatch tables rather than ELF relocations.
    pub retained_symbols: Vec<String>,
    pub externals: HashMap<String, u32>,
}

#[derive(Clone, Debug)]
pub struct LinkedImage {
    pub base: u32,
    pub entry: u32,
    pub bytes: Vec<u8>,
    pub symbols: BTreeMap<String, u32>,
}

#[derive(Clone, Debug)]
struct PlacedSection {
    object_index: usize,
    section_index: object::SectionIndex,
    address: u32,
    offset: usize,
    size: usize,
}

pub fn link(objects: &[Vec<u8>], options: &LinkOptions) -> Result<LinkedImage, ElfError> {
    info!(
        objects = objects.len(),
        base = format_args!("0x{:x}", options.image_base),
        entry = %options.entry_symbol,
        "linking tile image"
    );
    let parsed = objects
        .iter()
        .map(|bytes| object::File::parse(bytes.as_slice()))
        .collect::<Result<Vec<_>, _>>()?;
    let roots = std::iter::once(options.entry_symbol.as_str())
        .chain(options.retained_symbols.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let kept = reachable_sections(&parsed, &roots)?;
    debug!(sections = kept.len(), "retained reachable sections");
    let mut placements = Vec::new();
    let mut cursor = 0usize;
    for (object_index, file) in parsed.iter().enumerate() {
        for section in file.sections() {
            if !kept.contains(&(object_index, section.index())) {
                continue;
            }
            if !matches!(
                section.kind(),
                SectionKind::Text
                    | SectionKind::Data
                    | SectionKind::ReadOnlyData
                    | SectionKind::ReadOnlyString
                    | SectionKind::UninitializedData
            ) || section.size() == 0
            {
                continue;
            }
            let alignment = usize::try_from(section.align().max(1))
                .map_err(|_| ElfError::Link("section alignment overflow".into()))?;
            cursor = align_image_offset(options.image_base, cursor, alignment)?;
            let size = usize::try_from(section.size())
                .map_err(|_| ElfError::Link("section size overflow".into()))?;
            let address = options
                .image_base
                .checked_add(cursor as u32)
                .ok_or_else(|| ElfError::Link("image address overflow".into()))?;
            placements.push(PlacedSection {
                object_index,
                section_index: section.index(),
                address,
                offset: cursor,
                size,
            });
            cursor = cursor
                .checked_add(size)
                .ok_or_else(|| ElfError::Link("image size overflow".into()))?;
        }
    }
    let mut image = vec![0u8; cursor];
    for placement in &placements {
        let section = parsed[placement.object_index].section_by_index(placement.section_index)?;
        if section.kind() != SectionKind::UninitializedData {
            let data = section.uncompressed_data()?;
            if data.len() != placement.size {
                return Err(ElfError::Link("section data size mismatch".into()));
            }
            image[placement.offset..placement.offset + placement.size].copy_from_slice(&data);
        }
    }

    let mut symbols = BTreeMap::new();
    for (object_index, file) in parsed.iter().enumerate() {
        for symbol in file.symbols() {
            if !symbol.is_definition() || symbol.name().unwrap_or_default().is_empty() {
                continue;
            }
            let Some(section_index) = symbol.section_index() else {
                continue;
            };
            let Ok(placement) = placement(&placements, object_index, section_index) else {
                continue;
            };
            let value = placement
                .address
                .checked_add(symbol.address() as u32)
                .ok_or_else(|| ElfError::Link("symbol address overflow".into()))?;
            let name = symbol.name()?.to_owned();
            if symbols.insert(name.clone(), value).is_some() {
                return Err(ElfError::Link(format!("duplicate symbol {name}")));
            }
        }
    }
    for (name, value) in &options.externals {
        symbols.insert(name.clone(), *value);
    }
    debug!(?symbols, "resolved linked symbols");

    for (object_index, file) in parsed.iter().enumerate() {
        for section in file.sections() {
            let Some(place) = placements.iter().find(|placed| {
                placed.object_index == object_index && placed.section_index == section.index()
            }) else {
                continue;
            };
            for (offset, relocation) in section.relocations() {
                let target = match relocation.target() {
                    RelocationTarget::Symbol(index) => {
                        let symbol = file.symbol_by_index(index)?;
                        if symbol.is_undefined() {
                            *symbols.get(symbol.name()?).ok_or_else(|| {
                                ElfError::Link(format!(
                                    "undefined symbol {}",
                                    symbol.name().unwrap_or("?")
                                ))
                            })? as i64
                        } else {
                            let target_section = symbol.section_index().ok_or_else(|| {
                                ElfError::Link("absolute symbol unsupported".into())
                            })?;
                            let target_place =
                                placement(&placements, object_index, target_section)?;
                            i64::from(target_place.address) + symbol.address() as i64
                        }
                    }
                    RelocationTarget::Section(index) => {
                        i64::from(placement(&placements, object_index, index)?.address)
                    }
                    other => {
                        return Err(ElfError::Link(format!(
                            "unsupported relocation target {other:?}"
                        )));
                    }
                };
                let value = target
                    .checked_add(relocation.addend())
                    .ok_or_else(|| ElfError::Link("relocation value overflow".into()))?;
                if value < 0 {
                    return Err(ElfError::Link("negative relocation value".into()));
                }
                let location = place
                    .offset
                    .checked_add(offset as usize)
                    .ok_or_else(|| ElfError::Link("relocation offset overflow".into()))?;
                let object::RelocationFlags::Elf { r_type } = relocation.flags() else {
                    return Err(ElfError::Link("non-ELF relocation".into()));
                };
                apply_relocation(
                    &mut image,
                    location,
                    r_type,
                    value as u64,
                    options.image_base,
                )?;
            }
        }
    }
    let entry = *symbols
        .get(&options.entry_symbol)
        .ok_or_else(|| ElfError::Link(format!("missing entry symbol {}", options.entry_symbol)))?;
    let linked = LinkedImage {
        base: options.image_base,
        entry,
        bytes: image,
        symbols,
    };
    info!(
        bytes = linked.bytes.len(),
        symbols = linked.symbols.len(),
        entry = format_args!("0x{:x}", linked.entry),
        "tile image linked"
    );
    Ok(linked)
}

fn reachable_sections(
    objects: &[object::File<'_>],
    root_symbols: &[&str],
) -> Result<HashSet<(usize, object::SectionIndex)>, ElfError> {
    let mut definitions = HashMap::new();
    for (object_index, file) in objects.iter().enumerate() {
        for symbol in file.symbols() {
            if !symbol.is_definition() || !(symbol.is_global() || symbol.is_weak()) {
                continue;
            }
            if let (Ok(name), Some(section)) = (symbol.name(), symbol.section_index())
                && definitions
                    .insert(name.to_owned(), (object_index, section))
                    .is_some()
            {
                return Err(ElfError::Link(format!("duplicate symbol {name}")));
            }
        }
    }
    let mut kept = HashSet::new();
    let mut pending = root_symbols
        .iter()
        .map(|symbol| {
            definitions
                .get(*symbol)
                .copied()
                .ok_or_else(|| ElfError::Link(format!("missing retained symbol {symbol}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    while let Some((object_index, section_index)) = pending.pop() {
        if !kept.insert((object_index, section_index)) {
            continue;
        }
        let file = &objects[object_index];
        let section = file.section_by_index(section_index)?;
        for (_, relocation) in section.relocations() {
            let target = match relocation.target() {
                RelocationTarget::Section(index) => Some((object_index, index)),
                RelocationTarget::Symbol(index) => {
                    let symbol = file.symbol_by_index(index)?;
                    if let Some(index) = symbol.section_index() {
                        Some((object_index, index))
                    } else {
                        definitions.get(symbol.name()?).copied()
                    }
                }
                _ => None,
            };
            if let Some(target) = target {
                pending.push(target);
            }
        }
    }
    Ok(kept)
}

fn placement(
    placements: &[PlacedSection],
    object_index: usize,
    section_index: object::SectionIndex,
) -> Result<&PlacedSection, ElfError> {
    placements
        .iter()
        .find(|placement| {
            placement.object_index == object_index && placement.section_index == section_index
        })
        .ok_or_else(|| ElfError::Link("symbol refers to discarded section".into()))
}

fn align(value: usize, alignment: usize) -> Result<usize, ElfError> {
    if !alignment.is_power_of_two() {
        return Err(ElfError::Link("non-power-of-two section alignment".into()));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| ElfError::Link("alignment overflow".into()))
}

fn align_image_offset(base: u32, offset: usize, alignment: usize) -> Result<usize, ElfError> {
    let address = usize::try_from(base)
        .map_err(|_| ElfError::Link("image base overflow".into()))?
        .checked_add(offset)
        .ok_or_else(|| ElfError::Link("image address overflow".into()))?;
    align(address, alignment)?
        .checked_sub(base as usize)
        .ok_or_else(|| ElfError::Link("aligned image address precedes base".into()))
}

pub fn apply_relocation(
    image: &mut [u8],
    offset: usize,
    relocation: u32,
    mut value: u64,
    image_base: u32,
) -> Result<(), ElfError> {
    let invalid = |value| {
        ElfError::Link(format!(
            "unrepresentable relocation type {relocation} value 0x{value:x}"
        ))
    };
    match relocation {
        R_COLOSSUS_NONE => return Ok(()),
        R_COLOSSUS_8 => {
            if value > u8::MAX as u64 || offset >= image.len() {
                return Err(invalid(value));
            }
            image[offset] = value as u8;
            return Ok(());
        }
        R_COLOSSUS_16
        | R_COLOSSUS_16_S3
        | R_COLOSSUS_16_S4
        | R_COLOSSUS_16_S5
        | R_COLOSSUS_RELATIVE_16_S2 => {
            let shift = match relocation {
                R_COLOSSUS_RELATIVE_16_S2 => {
                    value = value
                        .checked_sub(u64::from(image_base))
                        .ok_or_else(|| invalid(value))?;
                    2
                }
                R_COLOSSUS_16_S3 => 3,
                R_COLOSSUS_16_S4 => 4,
                R_COLOSSUS_16_S5 => 5,
                _ => 0,
            };
            if value & ((1u64 << shift) - 1) != 0 {
                return Err(invalid(value));
            }
            value >>= shift;
            if value > u16::MAX as u64 {
                return Err(invalid(value));
            }
            write(image, offset, &(value as u16).to_le_bytes())?;
        }
        R_COLOSSUS_32 => {
            if value > u32::MAX as u64 {
                return Err(invalid(value));
            }
            write(image, offset, &(value as u32).to_le_bytes())?;
        }
        R_COLOSSUS_64 => write(image, offset, &value.to_le_bytes())?,
        R_COLOSSUS_18_S2 | R_COLOSSUS_19_S2 => {
            if value & 3 != 0 {
                return Err(invalid(value));
            }
            value >>= 2;
            let bits = if relocation == R_COLOSSUS_18_S2 {
                18
            } else {
                19
            };
            if value >= (1 << bits) {
                return Err(invalid(value));
            }
            let current = read_u32(image, offset)?;
            let mask = (1u32 << bits) - 1;
            write(
                image,
                offset,
                &((current & !mask) | value as u32).to_le_bytes(),
            )?;
        }
        R_COLOSSUS_20 | R_COLOSSUS_21 => {
            let bits = if relocation == R_COLOSSUS_20 { 20 } else { 21 };
            if value >= (1 << bits) {
                return Err(invalid(value));
            }
            let current = read_u32(image, offset)?;
            let mask = (1u32 << bits) - 1;
            write(
                image,
                offset,
                &((current & !mask) | value as u32).to_le_bytes(),
            )?;
        }
        R_COLOSSUS_RUN => {
            if value & 3 != 0 {
                return Err(invalid(value));
            }
            value = value
                .checked_sub(u64::from(image_base))
                .ok_or_else(|| invalid(value))?
                >> 2;
            if value > u16::MAX as u64 {
                return Err(invalid(value));
            }
            let current = read_u32(image, offset)?;
            let encoded = ((value as u32 & 0xf000) << 4) | (value as u32 & 0xfff);
            write(
                image,
                offset,
                &((current & 0xfff0f000) | encoded).to_le_bytes(),
            )?;
        }
        _ => {
            return Err(ElfError::Link(format!(
                "unknown Colossus relocation {relocation}"
            )));
        }
    }
    Ok(())
}

fn read_u32(image: &[u8], offset: usize) -> Result<u32, ElfError> {
    let bytes: [u8; 4] = image
        .get(offset..offset + 4)
        .ok_or_else(|| ElfError::Link("relocation outside section".into()))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(bytes))
}

fn write(image: &mut [u8], offset: usize, bytes: &[u8]) -> Result<(), ElfError> {
    let output = image
        .get_mut(offset..offset + bytes.len())
        .ok_or_else(|| ElfError::Link("relocation outside section".into()))?;
    output.copy_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_colossus_fields() {
        let mut bytes = 0xfff0_0000u32.to_le_bytes().to_vec();
        apply_relocation(&mut bytes, 0, R_COLOSSUS_20, 0xabcde, 0x4c000).unwrap();
        assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), 0xfffa_bcde);

        let mut bytes = 0xfff8_0000u32.to_le_bytes().to_vec();
        apply_relocation(&mut bytes, 0, R_COLOSSUS_19_S2, 0x50120, 0x4c000).unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes.try_into().unwrap()) & 0x7ffff,
            0x14048
        );
    }

    #[test]
    fn run_relocation_is_image_relative() {
        let mut bytes = 0xfff0_f000u32.to_le_bytes().to_vec();
        apply_relocation(&mut bytes, 0, R_COLOSSUS_RUN, 0x50000, 0x4c000).unwrap();
        let word = u32::from_le_bytes(bytes.try_into().unwrap());
        assert_eq!(word & 0x000f_0fff, 0x1_0000);
    }

    #[test]
    fn section_alignment_uses_absolute_image_address() {
        assert_eq!(align_image_offset(0x4c014, 0xc8, 16).unwrap(), 0xcc);
        assert_eq!(0x4c014 + 0xcc, 0x4c0e0);
        assert_eq!(align_image_offset(0x4c014, 0xcc, 16).unwrap(), 0xcc);
    }
}
