//! Search decisions and neighborhood enumeration. Candidate construction is
//! migrating here from mid; this owner never invokes package evaluation.
pub(crate) mod checkpoint;
mod proposals;
pub(crate) use proposals::proposals;
