//! Address-independent callable identities shared by construction, binding,
//! specialization and costing. Algorithm construction lives in the planner.

use serde::{Deserialize, Serialize};

/// Matrix axes and logical arithmetic bounds for a selected GEMM invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmAxes {
    pub left_inner: crate::TensorAxis,
    pub right_inner: crate::TensorAxis,
    pub output_column: crate::TensorAxis,
    pub valid_inner: Option<u32>,
    pub valid_columns: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccumulationPrecision {
    F16,
    F32,
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
