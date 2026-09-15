//! Portable search decisions, without physical addresses or scheduler caches.
use crate::compile::PipelineConfig;
use crate::graph::ComputeGraph;
use crate::package::{PackageBuildResult, invalid};
use crate::planner::{Candidate, Recipe};
use std::collections::BTreeMap;
use std::io::Write;

const VERSION: u32 = 6;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct State {
    version: u32,
    context: String,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<crate::OperationId, Vec<crate::planner::operator::OperatorPlan>>,
    pub inputs: BTreeMap<crate::ValueId, crate::TensorFormat>,
    pub visited: Vec<Recipe>,
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
            let recipe = match mapping {
                Some(mapping) => Recipe::default().remapped(graph, mapping, config.tile_count)?,
                None => Recipe::default(),
            };
            return Ok(Self {
                version: VERSION,
                context,
                recipe,
                ..Self::default()
            });
        };
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
                invalid(format!("invalid search state {}: {error}", path.display()))
            })?;
        if saved["version"] != VERSION {
            return Err(invalid(format!(
                "unsupported search-state version {}; rerun without --load-search-state",
                saved["version"]
            )));
        }
        let mut state: Self = serde_json::from_value(saved).map_err(|error| {
            invalid(format!("invalid search state {}: {error}", path.display()))
        })?;
        if state.context != context {
            return Err(invalid(format!(
                "search state does not match this graph/configuration: {}; rerun without --load-search-state",
                context_difference(&state.context, &context)
            )));
        }
        state.recipe.normalize(config);
        for recipe in &mut state.visited {
            recipe.normalize(config);
        }
        tracing::info!(path = %path.display(), attempts = state.attempts, "loaded mid-plan search state");
        Ok(state)
    }

    pub fn save(
        &mut self,
        config: &PipelineConfig,
        incumbent: &Candidate,
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

/// Keep checkpoint failures actionable without printing a whole graph or
/// concealing a configuration difference behind one opaque equality check.
fn context_difference(saved: &str, current: &str) -> String {
    let offset = saved
        .chars()
        .zip(current.chars())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| saved.chars().count().min(current.chars().count()));
    let start = offset.saturating_sub(48);
    let excerpt = |s: &str| s.chars().skip(start).take(144).collect::<String>();
    format!(
        "context differs at character {offset}; saved {:?}; current {:?}",
        excerpt(saved),
        excerpt(current)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_requires_current_schema_and_matching_context() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [4, 16]).unwrap();
        let mut config = PipelineConfig::new(4).with_automatic_input(input, crate::Precision::F16);
        let mut state = State::load(&graph, &config, None).unwrap();
        state.attempts = 17;
        let path = std::env::temp_dir().join(format!("ipu-checkpoint-{}.json", fastrand::u64(..)));
        config.load_search_state = Some(path.clone());
        let write = |value: &serde_json::Value| {
            std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        };
        let mut saved = serde_json::to_value(state).unwrap();
        for version in (0..VERSION).chain([VERSION + 1]) {
            write(&serde_json::json!({"version": version}));
            let error = State::load(&graph, &config, None)
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains("rerun without --load-search-state"),
                "{error}"
            );
        }
        write(&saved);
        assert_eq!(State::load(&graph, &config, None).unwrap().attempts, 17);
        config.optimization_steps += 3;
        assert!(State::load(&graph, &config, None).is_ok());
        config.profiling = !config.profiling;
        assert!(State::load(&graph, &config, None).is_err());
        config.profiling = !config.profiling;
        assert!(State::load(&graph, &config, Some(&[0, 1, 2, 3])).is_err());
        saved["context"] = "obsolete graph representation".into();
        write(&saved);
        assert!(State::load(&graph, &config, None).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
