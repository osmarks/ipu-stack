//! Address-resolved programs executed by individual tiles and the host.

use serde::{Deserialize, Serialize};

/// A fully resolved program for one logical tile.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileProgram {
    pub tile: u16,
    pub steps: Vec<TileStep>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileStep {
    Exchange(ExchangeStep),
    Compute(ComputeStep),
    Repeat(RepeatStep),
    Checkpoint(CheckpointStep),
}

/// A debugger-visible operator boundary using alternating PBRK0/PBRK1 traps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointStep {
    pub operation: u32,
    pub breakpoint: u8,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatStep {
    pub count: u32,
    /// Mutable bases used by [`TileAddress::RepeatPointer`] in the body.
    pub iterated_pointers: Vec<RepeatPointer>,
    pub body: Vec<TileStep>,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepeatPointer {
    pub initial_address: u32,
    pub stride_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileAddress {
    Absolute(u32),
    /// The current base of an enclosing repeat plus a constant byte offset.
    RepeatPointer {
        index: u16,
        offset: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeStep {
    /// Whether this tile executes a timed send/receive program after the boundary.
    pub active: bool,
    /// Base address used by point-to-point receive rows.
    pub incoming_base: u32,
    /// Preserve both exchange base registers on entry. Absolute-address paired
    /// rows use the two PIC streams directly and must not reset their state.
    #[serde(default)]
    pub preserve_base_registers: bool,
    /// Ordinary receive source selected outside the timed row when a paired
    /// receive uses the neighbouring sender for its waiting half.
    #[serde(default)]
    pub incoming_mux: Option<u16>,
    /// IPU21 incoming item format: 0 for 32-bit, 1 for the early half of a
    /// paired 64-bit path, and 2 for the waiting half.
    #[serde(default)]
    pub incoming_format: u8,
    /// Fixed source selection for the borrowed half of a paired 64-bit path.
    #[serde(default)]
    pub incoming_mux_pair: Option<u16>,
    /// Override the ordinary internal-exchange down-count. Paired 64-bit
    /// helper tiles execute mux timing while using zero to ignore the value.
    #[serde(default)]
    pub incoming_dcount: Option<u32>,
    /// The exchange row owns its supervisor sync and does not require the
    /// generic down-count setup. This is used by paired-width rows whose SDK
    /// form treats the sync and the following timing program as one unit.
    #[serde(default)]
    pub sync_in_program: bool,
    /// Synchronization-free timed exchange program.
    pub program: PlacedExchangeRow,
    /// Address words applied before invoking a structurally shared row.
    #[serde(default)]
    pub setup_patch: Option<ExchangeSetupPatch>,
    /// Words rewritten before the timed program is invoked inside a structured repeat.
    #[serde(default)]
    pub repeat_patches: Vec<ExchangePatch>,
    #[serde(default)]
    pub profile: StepProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangeSetupPatch {
    /// Byte offsets into the shared executable row, reused by its structural shape.
    pub offsets: PlacedExchangeRow,
    /// Replacement instruction words for this use of the row.
    pub values: PlacedExchangeRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExchangePatch {
    pub word_offset: u32,
    /// Full replacement instruction words, indexed by repeat iteration.
    pub values: PlacedExchangeRow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeStep {
    /// Exact linked kernel symbol; no naming convention is applied.
    pub symbol: String,
    pub output_address: TileAddress,
    pub input_addresses: Vec<TileAddress>,
    pub arguments: Vec<u32>,
    #[serde(default)]
    pub profile: StepProfile,
}

/// Optional explicit cycle-counter destinations around a step.
///
/// The addresses belong to caller-managed tile memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepProfile {
    pub before: Option<u32>,
    pub after: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPhase {
    pub address: u32,
    pub active: bool,
    pub run_table: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostProgram {
    pub initialize: Vec<HostPhase>,
    pub inputs: Vec<HostPhase>,
    pub outputs: Vec<HostPhase>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacedExchangeRow {
    pub address: u32,
    pub words: Vec<u32>,
}
