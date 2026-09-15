//! Byte-copy descriptions and explicit movement requests, before placement.
//! Mid copies carry these policies into low::expand::copy, which constructs
//! their movement. kernel::copy coalesces and binds the local byte-copy launches.

/// Requested realization of a whole-device coordinate copy. Explicit requests
/// are checked by movement lowering; Automatic selects from the actual geometry.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum CopyPolicy {
    #[default]
    Automatic,
    /// Run a rearrangement kernel on corresponding resident shards.
    LocalKernel,
    /// Transfer compatible physical spans directly to their destination.
    DirectRetile,
    /// Transfer logical values through row-major staging, then pack locally.
    StageLogicalThenTransform,
}

pub fn default_copy_policy(from: &crate::Layout, to: &crate::Layout) -> CopyPolicy {
    if from.order == to.order {
        CopyPolicy::DirectRetile
    } else {
        CopyPolicy::StageLogicalThenTransform
    }
}

/// Destination preparation for a selected copy. This is separate from its
/// logical/physical traversal policy; changing it does not change tensor values.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PackingPolicy {
    #[default]
    Automatic,
    /// Use direct word movement without destination packing scratch.
    Direct,
    /// Populate row-major scratch, then pack into the destination.
    Staged,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyOperation<Buffer> {
    pub source: Buffer,
    pub source_offset: u32,
    pub destination: Buffer,
    pub destination_offset: u32,
    pub bytes: u32,
    pub pattern: CopyPattern,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CopyPattern {
    #[default]
    Contiguous,
    Strided {
        rows: u32,
        row_bytes: u32,
        source_stride: u32,
        destination_stride: u32,
    },
}
