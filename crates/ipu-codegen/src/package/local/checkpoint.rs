//! Portable search decisions, without physical addresses or scheduler caches.
use super::*;
use std::collections::BTreeMap;
use std::io::Write;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(super) struct State {
    version: u32,
    context: String,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<crate::OperationId, Vec<crate::OperatorPlan>>,
    pub inputs: BTreeMap<crate::ValueId, crate::TensorFormat>,
    pub visited: Vec<Recipe>,
    pub mapping: Option<Vec<u16>>,
    pub mapping_checked: bool,
    pub attempts: usize,
}

impl State {
    pub fn load(
        graph: &ComputeGraph,
        config: &PipelineConfig,
        mapping: Option<&[u16]>,
    ) -> PackageBuildResult<Self> {
        // Keep the complete context rather than relying on a hash collision or
        // serde support for the compiler's graph/configuration implementation.
        // Diagnostics and the per-invocation budget do not change the search.
        let mut normalized = config.clone();
        normalized.optimization_steps = 0;
        normalized.load_search_state = None;
        normalized.save_search_state = None;
        normalized.memory_profile_directory = None;
        normalized.exchange_diagnostics = false;
        let context = format!("{graph:?}\n{normalized:?}\n{mapping:?}");
        let Some(path) = &config.load_search_state else {
            return Ok(Self {
                version: 1,
                context,
                mapping: mapping.map(<[u16]>::to_vec),
                ..Self::default()
            });
        };
        let state: Self = serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
            invalid(format!("invalid search state {}: {error}", path.display()))
        })?;
        if state.version != 1 || state.context != context {
            return Err(invalid(
                "search state does not match this graph/configuration or schema",
            ));
        }
        tracing::info!(path = %path.display(), attempts = state.attempts, "loaded mid-plan search state");
        Ok(state)
    }

    pub fn save(
        &mut self,
        config: &PipelineConfig,
        incumbent: &Baseline,
        fixed: &PipelineConfig,
    ) -> PackageBuildResult<()> {
        let Some(path) = &config.save_search_state else {
            return Ok(());
        };
        self.recipe = incumbent.recipe.clone();
        self.alternatives = incumbent.alternatives.clone();
        self.inputs = fixed.inputs.clone();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let bytes = serde_json::to_vec(self).map_err(|e| invalid(e.to_string()))?;
        let result = (|| -> std::io::Result<()> {
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            std::fs::rename(&temporary, path)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result?;
        tracing::info!(path = %path.display(), attempts = self.attempts, bytes = bytes.len(), "saved mid-plan search state");
        Ok(())
    }
}
