//! Address-independent callable identities shared by construction, binding,
//! specialization and costing. Distributed contraction axes live in mid Product.

use serde::{Deserialize, Serialize};

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
