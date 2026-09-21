//! Search-effort limits. Memory capacity comes from PipelineConfig and is
//! checked using MemoryPeaks on each fragment plus its live boundary context.
//! There is no independently assigned fixed/temporary budget per operation.

/// Applied after dominance pruning within identical live boundary states.
/// None retains the complete frontier; positive limits make search approximate.
/// These do not limit local candidate enumeration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchLimits {
    pub states_per_boundary: Option<usize>,
    pub paths_per_state: Option<usize>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            states_per_boundary: Some(128),
            paths_per_state: Some(16),
        }
    }
}
