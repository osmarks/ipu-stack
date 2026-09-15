//! Address-independent callable identities shared by construction, binding,
//! specialization and costing. Distributed contraction axes live in mid Product.

use crate::graph::AttentionOptions;
use crate::tensor::{Layout, Precision};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
}

/// A tile-local callable selected by a whole-device operator plan.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileKernelSpec {
    /// Physical byte range initialized during tile expansion.
    FillZero {
        offset: u32,
        bytes: u32,
        /// Copy mappings cover every logical element; only padding needs initialization.
        padding_only: bool,
    },
    Gemm {
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
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmKernelMode {
    Initialize,
    Accumulate,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GemmWeightLoad {
    Standard,
    Interleaved,
}
