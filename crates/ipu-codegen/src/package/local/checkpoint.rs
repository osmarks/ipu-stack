//! Portable search decisions, without physical addresses or scheduler caches.
use super::*;
use std::collections::BTreeMap;
use std::io::Write;

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(in crate::package) struct State {
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
        let mut state: Self = serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
            invalid(format!("invalid search state {}: {error}", path.display()))
        })?;
        // Version-one contexts included the removed deferred-view bookkeeping
        // field in concrete catalogue entries. It was never a configuration
        // choice. Normalize only the config portion, preserving graph names
        // verbatim; recipe deserialization ignores the obsolete field too.
        if state.version == 1 && state.context != context {
            if let Some((graph_context, config_context)) = state.context.split_once('\n') {
                let migrated = format!(
                    "{graph_context}\n{}",
                    config_context.replace(", deferred_output: None", "")
                );
                if migrated == context {
                    state.context = migrated;
                    tracing::info!("migrated checkpoint context without deferred-view metadata");
                }
            }
        }
        if state.version != 1 || state.context != context {
            return Err(invalid(
                "search state does not match this graph/configuration or schema",
            ));
        }
        if !state.recipe.early_casts.is_empty()
            || state
                .visited
                .iter()
                .any(|recipe| !recipe.early_casts.is_empty())
        {
            // Old visits describe outer-input cast ordering, not choices on
            // the expanded graph. Preserve the incumbent and search budget.
            state.visited.clear();
            tracing::info!("migrating legacy cast choices; cleared obsolete search visits");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_deferred_metadata_does_not_invalidate_search_decisions() {
        let mut graph = ComputeGraph::new();
        let input = graph
            .host_input("input, deferred_output: None", [4, 16])
            .unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(4).with_automatic_input(input, crate::Precision::F16);
        let mut state = State::load(&graph, &config, None).unwrap();
        let selected = baseline::lower(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::mid::implementation::FragmentCache::default(),
            &Recipe::default(),
        )
        .unwrap();
        state.recipe = selected.recipe;
        state.attempts = 17;
        let start = state.context.find("OperatorPlan {").unwrap();
        let mut depth = 0;
        let end = state.context[start..]
            .char_indices()
            .find_map(|(index, character)| {
                match character {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(start + index);
                        }
                    }
                    _ => {}
                }
                None
            })
            .unwrap();
        state.context.insert_str(end - 1, ", deferred_output: None");
        let expected = state.recipe.clone();
        let mut json = serde_json::to_value(&state).unwrap();
        for plan in json["recipe"]["plans"]
            .as_object_mut()
            .unwrap()
            .values_mut()
        {
            plan.as_object_mut()
                .unwrap()
                .insert("deferred_output".into(), serde_json::Value::Null);
        }
        let path =
            std::env::temp_dir().join(format!("ipu-deferred-context-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        config.load_search_state = Some(path.clone());
        let resumed = State::load(&graph, &config, None).unwrap();
        assert!(resumed.recipe == expected);
        assert_eq!(resumed.attempts, 17);
        config.profiling = !config.profiling;
        assert!(State::load(&graph, &config, None).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
