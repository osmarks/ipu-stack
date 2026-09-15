//! Select implementations and directly construct executable whole-device mid.
//! Graph construction owns enumeration/ranking; family constructors receive a
//! selected choice and actual tensor types. No package evaluation occurs here.
mod attention;
pub(crate) mod bind;
pub(crate) mod build;
pub(crate) mod cache;
pub(crate) mod candidates;
pub(crate) mod catalogue;
mod error;
pub(crate) mod fragments;
mod gemm;
mod layernorm;
pub(crate) mod operator;
mod parameter_homes;
mod proposals;
mod recipe;
pub use catalogue::OperatorCandidate;
pub use error::{LoweringError, LoweringResult};
pub use operator::{GemmOrientation, LocalOperandStaging, OperatorFamily};
pub(crate) use proposals::proposals;
pub(crate) use recipe::{Candidate, Recipe};
#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests;
