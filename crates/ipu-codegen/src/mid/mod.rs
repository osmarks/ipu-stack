//! Executable whole-device copies, arithmetic and structured repetition.
//! Planner choices/configuration live outside this language. Mid owns binding,
//! composition and explicit transformations of already selected work.
use crate::estimate::MemoryPeaks;
use crate::graph::AttentionOptions;
use crate::graph::{GraphInputKind, OperationId, ValueId};
use crate::kernel::{AccumulationPrecision, GemmKernelMode, GemmWeightLoad};
use crate::tensor::{Layout, Precision};
use crate::tensor::{OwnerMap, TensorType};
use ipu_target::Target;
use std::collections::BTreeMap;
pub(crate) mod cast_order;
mod compute;
mod copy;
mod fragment;
mod grouping;
pub(crate) mod ownership;
mod packing;
pub(crate) mod rewrite;
mod validate;
pub use compute::*;
pub(crate) use copy::default_copy_policy;
pub(crate) use copy::independent_copy_prefix;
pub use copy::{CoordinateMapping, CopyPolicy, PackingPolicy};
pub(crate) use fragment::append_fragment;
pub use validate::ProgramError;

#[cfg(test)]
pub(crate) fn expand_tiles(
    target: Target,
    program: &MidGraph,
) -> crate::ExpansionResult<std::sync::Arc<crate::TileGraph>> {
    crate::low::expand::expand_tiles(target, program, true)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MidValueId(u32);

impl MidValueId {
    pub const fn index(self) -> u32 {
        self.0
    }

    pub const fn from_index(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidValue {
    pub id: MidValueId,
    /// Selected embedding of layout owners onto device tiles. Distribution and
    /// storage order remain in tensor_type; values can share this map cheaply.
    pub owners: OwnerMap,
    pub tensor_type: TensorType,
    /// Semantic value represented by this value; conversions retain the same
    /// origin. Region arguments also refer to their high-level argument ID.
    pub origin: ValueId,
    /// Values in the same group use the same logical-to-physical tile mapping.
    /// Structured iteration uses this to keep successive parameter blocks
    /// addressable by a single advancing base pointer.
    pub storage_group: MidValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidOperationKind {
    /// Produce a distributed value in its declared layout. Lowering may alias
    /// compatible storage per tile, but must preserve that layout's interpretation.
    Copy {
        mapping: CoordinateMapping,
        policy: CopyPolicy,
        packing: PackingPolicy,
    },
    /// Physical byte range initialized during tile expansion.
    FillZero {
        offset: u32,
        bytes: u32,
        /// Copy mappings cover every logical element; only padding needs initialization.
        padding_only: bool,
    },
    /// Product on the selected distributed operand/output regions. Local
    /// batch and panel traversal belongs to the GEMM kernel family; independent
    /// K partitions and their reduction are separate operations in this graph.
    Gemm {
        axes: crate::kernel::GemmAxes,
        multiply: Precision,
        accumulate: AccumulationPrecision,
        mode: GemmKernelMode,
        weights: GemmWeightLoad,
        inner_block: u32,
        output_columns: u32,
    },
    Gelu,
    BiasGelu,
    AddLayerNorm,
    LayerNorm,
    /// FP32 mean and variance for each feature shard.
    LayerNormMoments,
    /// Produce FP32 moments and an explicit F16 residual sum.
    AddLayerNormMoments,
    /// Combine equal-width feature shards' moments and apply normalization.
    LayerNormApply {
        parts: u16,
    },
    ReductionSum {
        partials: u16,
    },
    Add,
    FlashAttention {
        options: AttentionOptions,
        accumulate: AccumulationPrecision,
    },
    AttentionSoftmax {
        head_dimension: u32,
        key_columns: u32,
        padded_key_columns: u32,
    },
    AttentionMerge {
        value_dimension: u32,
        padded_value_dimension: u32,
        initial: bool,
        final_block: bool,
    },
    Cast {
        from: Precision,
        to: Precision,
    },
    Rearrange {
        from: Layout,
        to: Layout,
    },
    Repeat(MidRepeat),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidOperation {
    pub source: Option<OperationId>,
    pub inputs: Vec<MidValueId>,
    pub results: Vec<MidValueId>,
    pub kind: MidOperationKind,
    pub operands: Vec<OperandIndexing>,
    pub output_aliases: Vec<(usize, usize)>,
    /// Writable ranges relative to each resident output fragment; omitted
    /// outputs use the full fragment. Writes retain the allocation's strides.
    pub output_windows: Vec<OperandWindow>,
}

impl MidOperation {
    /// Values read in the enclosing region, including a repeat's parameter sequences.
    pub(crate) fn read_values(&self) -> impl Iterator<Item = &MidValueId> {
        let sequences = match &self.kind {
            MidOperationKind::Repeat(repeat) => repeat.iterated_inputs.as_slice(),
            _ => &[],
        };
        self.inputs.iter().chain(sequences.iter().flatten())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRegion {
    pub arguments: Vec<MidValueId>,
    pub operations: Vec<MidOperation>,
    pub yields: Vec<MidValueId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidRepeat {
    pub count: u32,
    pub carried_inputs: usize,
    pub invariant_inputs: usize,
    /// One normalized value list for each iterated body argument. Keeping the
    /// lists on the structured operation avoids unrolling layer parameters.
    pub iterated_inputs: Vec<Vec<MidValueId>>,
    pub body: MidRegion,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidInput {
    pub name: String,
    pub kind: GraphInputKind,
    pub value: MidValueId,
}

/// Executable whole-device tensor program. Families append their operations
/// directly during selection; low then enumerates the concrete shard work.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MidGraph {
    pub tile_count: u16,
    pub inputs: Vec<MidInput>,
    pub values: Vec<MidValue>,
    pub operations: Vec<MidOperation>,
    pub outputs: Vec<MidValueId>,
    pub estimated_cycles: u64,
    pub estimated_exchange_cycles: u64,
    pub peak_memory: MemoryPeaks,
}

impl MidGraph {
    pub(crate) fn refresh_estimates(&mut self, target: Target) -> Option<()> {
        self.validate().ok()?;
        let (cycles, peak) = crate::estimate::analyze_mid(target, self, &BTreeMap::new())?;
        self.estimated_cycles = cycles.total;
        self.estimated_exchange_cycles = cycles.exchange;
        self.peak_memory = peak;
        Some(())
    }
}
