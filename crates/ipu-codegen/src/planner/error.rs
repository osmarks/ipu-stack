//! Failures constructing a selected semantic graph as executable mid.

use crate::graph::{OperationId, ValueId};
use crate::tensor::LayoutError;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum LoweringError {
    #[error("cast-before-copy choice is unavailable at {0:?}")]
    UnavailableCastChoice(crate::mid::WorkSite),
    #[error("legacy cast choice has no constructed site: operation {0:?}, ordinal {1}")]
    UnknownLegacyCastChoice(Option<OperationId>, u32),
    #[error("selected operator implementation is invalid")]
    InvalidImplementation,
    #[error(transparent)]
    InvalidProgram(#[from] crate::mid::ProgramError),
    #[error("cannot write planner memory profile: {0}")]
    MemoryProfile(String),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Storage(#[from] crate::storage::StorageError),
    #[error("mid-level lowering requires a nonzero tile count")]
    EmptyTileGroup,
    #[error("no tensor type was supplied for graph input {0:?}")]
    MissingInputType(ValueId),
    #[error("graph has no stored shape for value {0:?}")]
    MissingShape(ValueId),
    #[error("operation {0:?} has no legal format candidate")]
    NoCandidate(OperationId),
    #[error(
        "GEMM operation {0:?} has per-batch right operands; only weights broadcast across every batch dimension are currently supported"
    )]
    UnsupportedGemmBatching(OperationId),
    #[error("internal lowering error: value {0:?} is unavailable")]
    UnknownValue(ValueId),
}

pub type LoweringResult<T> = std::result::Result<T, LoweringError>;
