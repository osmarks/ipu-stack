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
        let mut saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
                invalid(format!("invalid search state {}: {error}", path.display()))
            })?;
        if matches!(saved["version"].as_u64(), Some(1 | 2)) {
            migrate_cast_schema(&mut saved)?;
        }
        if saved["version"] == 3 {
            migrate_ownership_schema(&mut saved, graph, config.tile_count)?;
        }
        if saved["version"] == 4 {
            migrate_packing_schema(&mut saved)?;
        }
        if saved["version"] == 5 {
            migrate_grouping_schema(&mut saved)?;
        }
        let mut state: Self = serde_json::from_value(saved).map_err(|error| {
            invalid(format!("invalid search state {}: {error}", path.display()))
        })?;
        if state.version == VERSION
            && state.context != context
            && matches_legacy_context(graph, &state.context, &context)
        {
            state.context = context.clone();
            tracing::info!("migrated checkpoint context without redundant bookkeeping");
        }
        if state.version != VERSION || state.context != context {
            let detail = if state.version != VERSION {
                format!("unsupported schema version {}", state.version)
            } else {
                context_difference(&state.context, &context)
            };
            return Err(invalid(format!(
                "search state does not match this graph/configuration: {detail}"
            )));
        }
        if state.recipe.has_legacy_choices() || state.visited.iter().any(Recipe::has_legacy_choices)
        {
            // Ordinal and global visits cannot be compared to named
            // choices without rebuilding each plan. Keep the incumbent and
            // search budget; discard only these obsolete visit identities.
            state.visited.clear();
            tracing::info!("migrating legacy site choices; cleared obsolete search visits");
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

/// Old ordinals survive only until construction can resolve their names. The
/// former global storage switch becomes the effective default of a scoped policy.
fn migrate_cast_schema(saved: &mut serde_json::Value) -> PackageBuildResult<()> {
    let ordinal_sites = saved["version"] == 1;
    let recipe = |value: &mut serde_json::Value| -> PackageBuildResult<()> {
        let object = value
            .as_object_mut()
            .ok_or_else(|| invalid("checkpoint recipe is not an object"))?;
        if ordinal_sites && let Some(casts) = object.remove("cast_before_copies") {
            object.insert("legacy_cast_sites".into(), casts);
        }
        if let Some(storage) = object.remove("in_place_casts")
            && !storage.is_null()
        {
            use crate::mid::cast::{CastStorage, CastStoragePolicy};
            let reuse = storage
                .as_bool()
                .ok_or_else(|| invalid("in_place_casts is not a boolean"))?;
            let policy = CastStoragePolicy::new(if reuse {
                CastStorage::ReuseIfSmaller
            } else {
                CastStorage::Separate
            });
            if object
                .insert(
                    "cast_storage".into(),
                    serde_json::to_value(policy).map_err(|error| invalid(error.to_string()))?,
                )
                .is_some()
            {
                return Err(invalid(
                    "checkpoint mixes legacy and scoped cast-storage choices",
                ));
            }
        }
        Ok(())
    };
    recipe(&mut saved["recipe"])?;
    if let Some(visited) = saved["visited"].as_array_mut() {
        for value in visited {
            recipe(value)?;
        }
    }
    saved["version"] = 3.into();
    tracing::info!("migrating legacy cast choices to scoped policies");
    Ok(())
}

/// The former global mapping applied to the incumbent and every visited recipe.
/// Convert it into input homes and operator working domains, preserving scoped
/// overrides and legacy cast fields until construction resolves their sites.
fn migrate_ownership_schema(
    saved: &mut serde_json::Value,
    graph: &ComputeGraph,
    tile_count: u16,
) -> PackageBuildResult<()> {
    let object = saved
        .as_object_mut()
        .ok_or_else(|| invalid("checkpoint is not an object"))?;
    let mapping: Option<Vec<u16>> =
        serde_json::from_value(object.remove("mapping").unwrap_or_default())
            .map_err(|error| invalid(format!("invalid checkpoint tile mapping: {error}")))?;
    object.remove("mapping_checked");
    if let Some(mapping) = mapping {
        let remap = |recipe: &mut serde_json::Value| -> PackageBuildResult<()> {
            let mut owners: crate::mid::OwnerChoices = match recipe.get("owners") {
                Some(value) => serde_json::from_value(value.clone())
                    .map_err(|error| invalid(error.to_string()))?,
                None => crate::mid::OwnerChoices::default(),
            };
            owners.remap_tiles(
                graph.inputs().iter().map(|input| input.value),
                graph.walk_operations().map(|operation| operation.id),
                &mapping,
                tile_count,
            )?;
            recipe["owners"] =
                serde_json::to_value(owners).map_err(|error| invalid(error.to_string()))?;
            Ok(())
        };
        remap(&mut saved["recipe"])?;
        if let Some(visited) = saved["visited"].as_array_mut() {
            for recipe in visited {
                remap(recipe)?;
            }
        }
    }
    saved["version"] = 4.into();
    tracing::info!("migrated global tile mapping into scoped recipe ownership");
    Ok(())
}

fn migrate_packing_schema(saved: &mut serde_json::Value) -> PackageBuildResult<()> {
    let migrate = |recipe: &mut serde_json::Value| -> PackageBuildResult<()> {
        let object = recipe
            .as_object_mut()
            .ok_or_else(|| invalid("checkpoint recipe is not an object"))?;
        if let Some(rows) = object.remove("packing_rows") {
            if object.insert("legacy_packing_rows".into(), rows).is_some() {
                return Err(invalid("checkpoint has multiple global packing requests"));
            }
        }
        Ok(())
    };
    migrate(&mut saved["recipe"])?;
    if let Some(visited) = saved["visited"].as_array_mut() {
        for recipe in visited {
            migrate(recipe)?;
        }
    }
    saved["version"] = 5.into();
    tracing::info!("migrating global packing rows to named copy choices");
    Ok(())
}

fn migrate_grouping_schema(saved: &mut serde_json::Value) -> PackageBuildResult<()> {
    let migrate = |recipe: &mut serde_json::Value| -> PackageBuildResult<()> {
        let object = recipe
            .as_object_mut()
            .ok_or_else(|| invalid("checkpoint recipe is not an object"))?;
        for (old, new) in [
            ("parallel_reductions", "legacy_parallel_reductions"),
            ("disjoint_copy_sources", "legacy_disjoint_copy_sources"),
        ] {
            if let Some(value) = object.remove(old)
                && object.insert(new.into(), value).is_some()
            {
                return Err(invalid("checkpoint has multiple global grouping requests"));
            }
        }
        if let Some(owners) = object
            .get_mut("owners")
            .and_then(serde_json::Value::as_object_mut)
            && let Some(results) = owners.remove("results")
            && object
                .insert("legacy_result_bases".into(), results)
                .is_some()
        {
            return Err(invalid(
                "checkpoint has multiple legacy result-home requests",
            ));
        }
        Ok(())
    };
    migrate(&mut saved["recipe"])?;
    if let Some(visited) = saved["visited"].as_array_mut() {
        for recipe in visited {
            migrate(recipe)?;
        }
    }
    saved["version"] = VERSION.into();
    tracing::info!("migrating global grouping and result bases to named groups and actual homes");
    Ok(())
}

/// Version-one contexts recorded implementation details alongside the graph.
/// Accept only the exact former representation of the current graph: its value
/// registry duplicated the shape keys, and its sequence counter was the length
/// of the sequence list. Reconstructing those fields preserves all comparisons,
/// including shapes, names, nested regions and the redundant fields themselves.
fn matches_legacy_context(graph: &ComputeGraph, saved: &str, current: &str) -> bool {
    let Some((saved_graph, saved_config)) = saved.split_once('\n') else {
        return false;
    };
    let Some((current_graph, current_config)) = current.split_once('\n') else {
        return false;
    };
    // Concrete catalogue entries also contained removed deferred-view metadata.
    // This is deliberately restricted to the configuration, not graph names.
    if saved_config.replace(", deferred_output: None", "") != current_config {
        return false;
    }
    if saved_graph == current_graph {
        return true;
    }
    // The root shape table is followed only by integer counters. Work backwards
    // from there so metadata-looking text in an input name cannot be mistaken
    // for a graph field.
    let Some((prefix, tail)) = current_graph.rsplit_once(", shapes: ") else {
        return false;
    };
    let Some(tail) = tail.strip_suffix(" }") else {
        return false;
    };
    let values = graph
        .value_shapes()
        .keys()
        .collect::<std::collections::BTreeSet<_>>();
    saved_graph
        == format!(
            "{prefix}, values: {values:?}, shapes: {tail}, next_sequence: {} }}",
            graph.sequences().len()
        )
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
    use crate::estimate::Ipu21CostModel;

    #[test]
    fn grouping_checkpoint_migration_preserves_pending_requests() {
        let mut graph = crate::ComputeGraph::new();
        let x = graph.host_input("x", [4, 16]).unwrap();
        graph.gelu(x).unwrap();
        let site = crate::mid::ResultSite {
            work: crate::mid::WorkSite {
                source: graph.operations()[0].id,
                local: "gelu".into(),
            },
            result: 0,
        };
        let mut old_recipe = Recipe::default();
        old_recipe
            .owners
            .results
            .insert(site.clone(), crate::tensor::OwnerMap::rotated(3));
        let mut recipe = serde_json::to_value(&old_recipe).unwrap();
        recipe["parallel_reductions"] = 2.into();
        recipe["disjoint_copy_sources"] = true.into();
        let mut saved = serde_json::json!({"version":5, "recipe":recipe, "visited":[recipe]});
        migrate_grouping_schema(&mut saved).unwrap();
        for value in [&saved["recipe"], &saved["visited"][0]] {
            let migrated: Recipe = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(migrated.legacy_parallel_reductions, 2);
            assert!(migrated.legacy_disjoint_copy_sources);
            assert_eq!(
                migrated.legacy_result_bases[&site],
                old_recipe.owners.results[&site]
            );
            assert!(migrated.owners.results.is_empty());
        }
        assert_eq!(saved["version"], VERSION);
    }

    #[test]
    fn legacy_packing_becomes_scoped_and_layout_search_can_replace_it() {
        use crate::planner::catalogue::pointwise_operator_candidate;
        use crate::planner::operator::OutputAliasing;
        use crate::tensor::{BlockMajorOrder, ElementOrder, Layout, Precision, TensorFormat};

        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [512, 32]).unwrap();
        let z = graph.host_input("z", [512, 32]).unwrap();
        let a = graph.gelu(x).unwrap();
        let b = graph.gelu(z).unwrap();
        graph.set_outputs([a, b]).unwrap();
        let plain = TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(64),
        };
        let mut target = TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(2),
        };
        target.layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: 256,
            column_block: 16,
        });
        let plan = |format: TensorFormat| {
            pointwise_operator_candidate(
                crate::planner::OperatorFamily::Gelu,
                [format.clone()],
                format,
            )
            .with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0]))
            .plan
        };
        let mut config = PipelineConfig::new(64)
            .with_input(x, plain.clone())
            .with_input(z, plain.clone());
        let mut recipe = Recipe::default();
        recipe.open_boundaries.extend([a, b]);
        for operation in graph.operations() {
            recipe.plans.insert(operation.id, plan(target.clone()));
        }
        let fragments = crate::planner::FragmentCache::default();
        let baseline =
            crate::planner::build_candidate(&graph, &config, &Ipu21CostModel, &fragments, &recipe)
                .unwrap();
        assert_eq!(baseline.packing_choices.len(), 2);
        let mut scoped = baseline.recipe.clone();
        scoped.packing = baseline
            .packing_choices
            .iter()
            .map(|(site, choices)| {
                (
                    site.clone(),
                    choices
                        .iter()
                        .find(|choice| choice.rows.get() == 128)
                        .unwrap()
                        .clone(),
                )
            })
            .collect();
        let expected =
            crate::planner::build_candidate(&graph, &config, &Ipu21CostModel, &fragments, &scoped)
                .unwrap();
        let mut state = State::load(&graph, &config, None).unwrap();
        state.recipe = baseline.recipe;
        state.attempts = 17;
        let mut saved = serde_json::to_value(state).unwrap();
        saved["version"] = 4.into();
        saved["recipe"].as_object_mut().unwrap().remove("packing");
        saved["recipe"]["packing_rows"] = 128.into();
        saved["visited"] = serde_json::json!([saved["recipe"].clone()]);
        let path =
            std::env::temp_dir().join(format!("ipu-packing-migration-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        config.load_search_state = Some(path.clone());
        config.save_search_state = Some(path.clone());
        let mut resumed = State::load(&graph, &config, None).unwrap();
        assert_eq!(resumed.attempts, 17);
        assert!(resumed.visited.is_empty());
        let mut actual = crate::planner::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &resumed.recipe,
        )
        .unwrap();
        assert_eq!(actual.program, expected.program);
        assert!(actual.recipe == expected.recipe);
        resumed.save(&config, &actual, &config).unwrap();
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["version"], VERSION);
        assert!(saved["recipe"].get("packing_rows").is_none());
        assert!(saved["recipe"].get("legacy_packing_rows").is_none());
        assert!(State::load(&graph, &config, None).unwrap().recipe == actual.recipe);

        let source = graph.operations()[0].id;
        let alternative = plan(plain);
        actual
            .alternatives
            .insert(source, vec![alternative.clone()]);
        let proposals = crate::planner::proposals(&graph, &config, &actual, None);
        let changed = |recipe: &Recipe| recipe.plans.get(&source) == Some(&alternative);
        let preserved = proposals
            .iter()
            .find(|p| changed(&p.recipe) && p.recipe.packing == actual.recipe.packing)
            .unwrap();
        assert!(
            crate::planner::build_candidate(
                &graph,
                &config,
                &Ipu21CostModel,
                &fragments,
                &preserved.recipe
            )
            .err()
            .unwrap()
            .to_string()
            .contains("packing")
        );
        let cleared = proposals
            .iter()
            .find(|p| {
                changed(&p.recipe)
                    && p.recipe.packing.len() == 1
                    && p.recipe.packing.keys().all(|site| site.source != source)
            })
            .unwrap();
        crate::planner::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &cleared.recipe,
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_mapping_migrates_incumbent_and_visits_into_owner_choices() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [32, 64]).unwrap();
        let y = graph.gelu(x).unwrap();
        graph.set_outputs([y]).unwrap();
        let mut config = PipelineConfig::new(8).with_automatic_input(x, crate::Precision::F16);
        let fragments = crate::planner::FragmentCache::default();
        let baseline = crate::planner::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &Recipe::default(),
        )
        .unwrap();
        let mut state = State::load(&graph, &config, None).unwrap();
        state.recipe = baseline.recipe;
        state.attempts = 19;
        let mut saved = serde_json::to_value(state).unwrap();
        let mapping = vec![0, 4, 1, 5, 2, 6, 3, 7];
        saved["version"] = 3.into();
        saved["mapping"] = serde_json::json!(mapping);
        saved["mapping_checked"] = true.into();
        saved["recipe"].as_object_mut().unwrap().remove("owners");
        saved["visited"] = serde_json::json!([saved["recipe"].clone()]);
        let path =
            std::env::temp_dir().join(format!("ipu-owner-migration-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        config.load_search_state = Some(path.clone());
        let resumed = State::load(&graph, &config, None).unwrap();
        assert_eq!(resumed.attempts, 19);
        assert!(resumed.visited[0] == resumed.recipe);
        let actual = crate::planner::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &resumed.recipe,
        )
        .unwrap();
        let mut expected = baseline.program;
        expected.remap_tiles(&mapping).unwrap();
        assert_eq!(actual.program.values, expected.values);
        assert_eq!(actual.program.operations, expected.operations);
        let saved = serde_json::to_value(resumed).unwrap();
        assert!(saved.get("mapping").is_none());
        assert!(saved.get("mapping_checked").is_none());
        assert_eq!(saved["version"], VERSION);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_cast_storage_defaults_preserve_incumbent_and_visited_choices() {
        use crate::mid::cast::{CastStorage, CastStoragePolicy};
        for reuse in [false, true] {
            let mut saved = serde_json::to_value(State::default()).unwrap();
            saved["version"] = 2.into();
            saved["attempts"] = 17.into();
            saved["recipe"]
                .as_object_mut()
                .unwrap()
                .remove("cast_storage");
            saved["recipe"]["in_place_casts"] = reuse.into();
            saved["visited"] = serde_json::json!([saved["recipe"].clone()]);
            migrate_cast_schema(&mut saved).unwrap();
            let state: State = serde_json::from_value(saved).unwrap();
            let expected = CastStoragePolicy::new(if reuse {
                CastStorage::ReuseIfSmaller
            } else {
                CastStorage::Separate
            });
            assert_eq!(state.attempts, 17);
            assert_eq!(state.recipe.cast_storage, Some(expected.clone()));
            assert_eq!(state.visited[0].cast_storage, Some(expected));
            assert!(
                serde_json::to_value(state).unwrap()["recipe"]
                    .get("in_place_casts")
                    .is_none()
            );
        }
    }

    #[test]
    fn ordinal_cast_checkpoint_replays_and_saves_named_choices() {
        use crate::planner::{build_candidate, cache::FragmentCache};

        let mut graph = ComputeGraph::new();
        let q = graph.host_input("q", [4, 17, 72]).unwrap();
        let k = graph.host_input("k", [4, 73, 72]).unwrap();
        let v = graph.host_input("v", [4, 73, 72]).unwrap();
        let result = graph.flash_attention(q, k, v).unwrap();
        graph.set_outputs([result]).unwrap();
        let mut config = PipelineConfig::new(64)
            .with_attention_products(crate::compile::AttentionProducts::Independent)
            .with_automatic_input(q, crate::Precision::F16)
            .with_automatic_input(k, crate::Precision::F16)
            .with_automatic_input(v, crate::Precision::F16);
        config.attention_fp8_scales = [Some(-4), None];
        let fragments = FragmentCache::default();
        let late = build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &Recipe::default(),
        )
        .unwrap();
        let site = late.cast_sites.iter().next_back().unwrap().clone();
        let raw = crate::planner::build::select(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &late.recipe,
        )
        .unwrap();
        let ordinal = raw
            .program
            .operations
            .iter()
            .filter(|operation| operation.source == Some(site.source))
            .filter(|operation| {
                crate::mid::rewrite::fp8_cast(operation, &raw.program.values).is_some()
            })
            .position(|operation| operation.work_site().as_ref() == Some(&site))
            .unwrap();
        let mut named = late.recipe;
        named.cast_before_copies.insert(site);
        let expected =
            build_candidate(&graph, &config, &Ipu21CostModel, &fragments, &named).unwrap();
        let mut state = State::load(&graph, &config, None).unwrap();
        state.recipe = named;
        state.attempts = 29;
        let mut saved = serde_json::to_value(&state).unwrap();
        saved["version"] = 1.into();
        saved["recipe"]
            .as_object_mut()
            .unwrap()
            .remove("cast_storage");
        saved["recipe"]["in_place_casts"] = false.into();
        saved["recipe"]["cast_before_copies"] = serde_json::json!([[
            state.recipe.cast_before_copies.first().unwrap().source,
            ordinal
        ]]);
        saved["visited"] = serde_json::json!([saved["recipe"].clone()]);
        let path = std::env::temp_dir().join(format!("ipu-cast-sites-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&saved).unwrap()).unwrap();
        config.load_search_state = Some(path.clone());
        config.save_search_state = Some(path.clone());
        let mut resumed = State::load(&graph, &config, None).unwrap();
        assert_eq!(resumed.attempts, 29);
        assert!(resumed.visited.is_empty());
        let actual = build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &fragments,
            &resumed.recipe,
        )
        .unwrap();
        assert_eq!(actual.program, expected.program);
        assert!(actual.recipe == expected.recipe);
        resumed.save(&config, &actual, &config).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(saved["version"], VERSION);
        assert!(saved["recipe"].get("legacy_cast_sites").is_none());
        assert!(saved["recipe"].get("early_casts").is_none());
        assert!(saved["recipe"].get("in_place_casts").is_none());
        let replay = State::load(&graph, &config, None).unwrap();
        assert_eq!(replay.attempts, 29);
        assert!(replay.recipe == actual.recipe);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_graph_registry_preserves_graph_and_configuration_checks() {
        let name = "input, values: {ValueId(99)}, shapes: {}, next_sequence: 7";
        let make_graph = |name: &str, width| {
            let mut graph = ComputeGraph::new();
            let input = graph.host_input(name, [4, width]).unwrap();
            graph.value_sequence("sequence", [input]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            graph
        };
        let graph = make_graph(name, 16);
        let mut config = PipelineConfig::new(4)
            .with_automatic_input(graph.inputs()[0].value, crate::Precision::F16);
        let mut state = State::load(&graph, &config, None).unwrap();
        state.attempts = 23;
        let (_, config_context) = state.context.split_once('\n').unwrap();
        // The old Debug representation, including both removed fields. Use
        // explicit expected counters and IDs rather than the migration helper.
        state.context = format!(
            "ComputeGraph {{ inputs: {:?}, sequences: {:?}, operations: {:?}, outputs: {:?}, values: {{ValueId(0), ValueId(1)}}, shapes: {:?}, next_operation: 1, next_value: 2, next_sequence: 1 }}\n{config_context}",
            graph.inputs(),
            graph.sequences(),
            graph.operations(),
            graph.outputs(),
            graph.value_shapes(),
        );
        let legacy = state.context.clone();
        let path =
            std::env::temp_dir().join(format!("ipu-graph-context-{}.json", std::process::id()));
        config.load_search_state = Some(path.clone());
        let write = |state: &State| {
            std::fs::write(&path, serde_json::to_vec(state).unwrap()).unwrap();
        };
        write(&state);
        let resumed = State::load(&graph, &config, None).unwrap();
        assert_eq!(resumed.attempts, 23);
        assert!(resumed.context.starts_with(&format!("{graph:?}\n")));
        assert!(State::load(&make_graph(name, 32), &config, None).is_err());
        assert!(State::load(&make_graph("different input", 16), &config, None).is_err());
        assert!(State::load(&graph, &config, Some(&[0, 1, 2, 3])).is_err());
        config.profiling = !config.profiling;
        assert!(State::load(&graph, &config, None).is_err());
        config.profiling = !config.profiling;
        for malformed in [
            legacy.replace("values: {ValueId(0), ValueId(1)}", "values: {ValueId(0)}"),
            legacy.replace("next_sequence: 1 }", "next_sequence: 2 }"),
        ] {
            state.context = malformed;
            write(&state);
            assert!(State::load(&graph, &config, None).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }

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
        let selected = crate::planner::build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
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
