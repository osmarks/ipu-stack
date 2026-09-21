//! Opt-in explanations of the planner's existing memory estimate.
use super::*;
use crate::mid::MidOperationKind;
use crate::{HighGraph, PipelineConfig};
use serde::Serialize;

#[derive(Serialize)]
struct Value {
    id: u32,
    origin: u32,
    root: usize,
    name: String,
    group: String,
    shape: Vec<u32>,
    precision: String,
    layout: String,
    layout_error: Option<String>,
    class: String,
    storage_group: u32,
    shard_bytes: u64,
    aligned_bytes: u64,
    copies: u32,
    bytes: u64,
    tile_shards: Vec<u64>,
    tile_bytes: Vec<u64>,
}

#[derive(Serialize)]
struct Step {
    index: usize,
    source: Option<u32>,
    operation: String,
    execution_count: u64,
    live: Vec<usize>,
    scratch: MemoryUsage,
    usage: MemoryUsage,
    tile_usage: Vec<[u64; 2]>,
    tile_scratch: Vec<[u64; 2]>,
    coarse_usage: MemoryUsage,
}

#[derive(Default, Serialize)]
struct Timeline {
    values: Vec<Value>,
    steps: Vec<Step>,
}

impl mid::MemoryObserver for Timeline {
    fn value(
        &mut self,
        value: &MidValue,
        root: usize,
        shard_bytes: u64,
        aligned_bytes: u64,
        copies: u32,
        bytes: u64,
        tile_shards: &[u64],
        tile_bytes: &[u64],
    ) {
        self.values.push(Value {
            id: value.id.index(),
            origin: value.origin.index(),
            root,
            name: String::new(),
            group: String::new(),
            shape: value.tensor_type.shape.0.clone(),
            precision: format!("{:?}", value.tensor_type.format.precision),
            layout: format!("{:?}", value.tensor_type.format.layout),
            layout_error: (shard_bytes == u64::MAX).then(|| {
                value
                    .tensor_type
                    .format
                    .layout
                    .resolve(&value.tensor_type.shape)
                    .err()
                    .map_or_else(|| "shard size overflow".into(), |error| error.to_string())
            }),
            class: format!("{:?}", value.tensor_type.format.layout.memory_class),
            storage_group: value.storage_group.index(),
            shard_bytes,
            aligned_bytes,
            copies,
            bytes,
            tile_shards: tile_shards.to_vec(),
            tile_bytes: tile_bytes.to_vec(),
        });
    }

    fn step(
        &mut self,
        index: usize,
        operation: &MidOperation,
        count: u64,
        live: &[bool],
        scratch: MemoryUsage,
        usage: MemoryUsage,
        tile_usage: &[MemoryUsage],
        tile_scratch: &[MemoryUsage],
    ) {
        let description = match &operation.kind {
            MidOperationKind::Copy {
                policy, packing, ..
            } => format!("Copy {policy:?} {packing:?}"),
            MidOperationKind::Repeat(repeat) => format!("Repeat {} boundary", repeat.count),
            kind => format!("{kind:?}"),
        };
        let mut roots = BTreeMap::new();
        for value in &self.values {
            if live[value.root] {
                let entry = roots.entry(value.root).or_insert((0, &value.class));
                entry.0 = entry.0.max(value.bytes);
            }
        }
        let mut coarse_usage = scratch;
        for (bytes, class) in roots.values() {
            if class.as_str() == "Ipu21Standard" {
                coarse_usage.standard += bytes;
            } else {
                coarse_usage.interleaved += bytes;
            }
        }
        self.steps.push(Step {
            index,
            source: operation.source.map(|id| id.index()),
            operation: description,
            execution_count: count,
            live: live
                .iter()
                .enumerate()
                .filter_map(|(id, &live)| live.then_some(id))
                .collect(),
            scratch,
            usage,
            tile_usage: tile_usage
                .iter()
                .map(|u| [u.standard, u.interleaved])
                .collect(),
            tile_scratch: tile_scratch
                .iter()
                .map(|u| [u.standard, u.interleaved])
                .collect(),
            coarse_usage,
        });
    }
}

fn names(graph: &HighGraph) -> BTreeMap<u32, (String, String)> {
    let mut names = graph
        .inputs()
        .iter()
        .map(|input| {
            (
                input.value.index(),
                (input.name.clone(), input.name.clone()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for sequence in graph.sequences() {
        for value in &sequence.values {
            if let Some((_, group)) = names.get_mut(&value.index()) {
                *group = sequence.name.clone();
            }
        }
    }
    fn visit(
        operations: &[crate::graph::Operation],
        graph: &HighGraph,
        names: &mut BTreeMap<u32, (String, String)>,
    ) {
        for operation in operations {
            for (index, value) in operation.results.iter().enumerate() {
                let name = format!("operation {} result {}", operation.id.index(), index);
                names.insert(value.index(), (name.clone(), name));
            }
            if let crate::graph::OperationKind::Repeat(repeat) = &operation.kind {
                for (index, argument) in repeat.body.arguments.iter().enumerate() {
                    let (name, group) = if index < operation.inputs.len() {
                        names
                            .get(&operation.inputs[index].index())
                            .cloned()
                            .unwrap_or_else(|| ("region argument".into(), "region argument".into()))
                    } else {
                        let sequence = &graph.sequences()[repeat.iterated_inputs
                            [index - operation.inputs.len()]
                        .index()
                            as usize];
                        (sequence.name.clone(), sequence.name.clone())
                    };
                    names.insert(argument.index(), (format!("{name} (body argument)"), group));
                }
                visit(&repeat.body.operations, graph, names);
            }
        }
    }
    visit(graph.operations(), graph, &mut names);
    names
}

#[derive(Serialize)]
struct Profile {
    version: u32,
    scope: String,
    model: &'static str,
    tile_count: u16,
    budget_bytes: u64,
    interleaved_budget_bytes: u64,
    standard_contiguous_overflow_bytes: u64,
    fits_budget: bool,
    support_reserve_bytes: u64,
    effective_peak_bytes: u64,
    peak: MemoryPeaks,
    total_peak_step: Option<usize>,
    standard_peak_step: Option<usize>,
    interleaved_peak_step: Option<usize>,
    timeline: Timeline,
}

fn profile(
    scope: &str,
    graph: &HighGraph,
    config: &PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    copies: &BTreeMap<MidValueId, u32>,
) -> Option<Profile> {
    let mut program = mid::region_program(config.tile_count, initial, operations, outputs, values);
    program.compose_copies();
    for input in &mut program.inputs {
        let origin = program.values[input.value.index() as usize].origin;
        if let Some(source) = graph.inputs().iter().find(|source| source.value == origin) {
            input.kind = source.kind;
        }
    }
    let mut timeline = Timeline::default();
    let (_, peak) = mid::analyze_observed(&program, copies, &mut timeline)?;
    let names = names(graph);
    for value in &mut timeline.values {
        (value.name, value.group) = names.get(&value.origin).cloned().unwrap_or_else(|| {
            (
                format!("value {}", value.origin),
                format!("value {}", value.origin),
            )
        });
    }
    let peak_step = |key: fn(&Step) -> u64| {
        timeline
            .steps
            .iter()
            .max_by_key(|step| (key(step), std::cmp::Reverse(step.index)))
            .map(|step| step.index)
    };
    Some(Profile {
        version: 3,
        scope: scope.into(),
        model: "Per-tile live storage using selected ownership, before address placement. Scratch remains estimated. Repeat execution counts do not multiply scratch. Exchange rows are reserved separately.",
        tile_count: config.tile_count,
        budget_bytes: config
            .tile_memory_budget_bytes
            .min(config.target.planned_data_bytes()),
        interleaved_budget_bytes: config.target.interleaved_data_bytes(),
        standard_contiguous_overflow_bytes: peak.standard_contiguous_overflow_with_reservation(
            config.target,
            config.standard_memory_reservation_bytes,
        ),
        fits_budget: peak.fits_with_budget(
            config.target,
            config.standard_memory_reservation_bytes,
            config.tile_memory_budget_bytes,
        ),
        support_reserve_bytes: config.standard_memory_reservation_bytes,
        effective_peak_bytes: peak
            .total
            .saturating_add(config.standard_memory_reservation_bytes),
        peak,
        total_peak_step: peak_step(|step| step.usage.total()),
        standard_peak_step: peak_step(|step| {
            step.tile_usage.iter().map(|u| u[0]).max().unwrap_or(0)
        }),
        interleaved_peak_step: peak_step(|step| {
            step.tile_usage.iter().map(|u| u[1]).max().unwrap_or(0)
        }),
        timeline,
    })
}

/// Called only for exhausted memory shortlists and selected planner finalists.
/// Diagnostics never force every candidate to retain a detailed timeline.
pub(crate) fn write(
    graph: &HighGraph,
    config: &PipelineConfig,
    program: &crate::MidGraph,
    scope: &str,
) -> crate::planner::PlanningResult<()> {
    let Some(directory) = &config.memory_profile_directory else {
        return Ok(());
    };
    let initial = program
        .inputs
        .iter()
        .map(|input| input.value)
        .collect::<Vec<_>>();
    let profile = profile(
        scope,
        graph,
        config,
        &initial,
        &program.operations,
        &program.outputs,
        &program.values,
        &Default::default(),
    )
    .ok_or(crate::planner::PlanningError::InvalidFragment(
        "memory profile",
    ))?;
    let result = (|| -> Result<_, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(directory)?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stem = format!("{scope}-{}-{serial:04}", std::process::id());
        let path = directory.join(stem).with_extension("json");
        let json = serde_json::to_string(&profile)?;
        std::fs::write(&path, &json)?;
        // Names are data, including graph inputs supplied by callers. Escaping
        // '<' keeps arbitrary names from terminating the embedded JSON script.
        let html = include_str!("memory_profile.html")
            .replace("__PROFILE_JSON__", &json.replace('<', "\\u003c"));
        std::fs::write(path.with_extension("html"), html)?;
        Ok(path)
    })()
    .map_err(|error| crate::planner::PlanningError::MemoryProfile(error.to_string()))?;
    tracing::info!(path = %result.display(), scope,
        peak_step = profile.total_peak_step, effective_peak_bytes = profile.effective_peak_bytes,
        "wrote planner memory profile");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_remain_live_after_their_last_operator_use() {
        let mut graph = HighGraph::new();
        let weight = graph.parameter("weight", [8, 64]).unwrap();
        let input = graph.host_input("input", [8, 64]).unwrap();
        let added = graph.add(input, weight).unwrap();
        let output = graph.gelu(added).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(4)
            .with_input(
                input,
                crate::TensorFormat {
                    precision: crate::Precision::F16,
                    layout: crate::Layout::row_sharded(4),
                },
            )
            .with_input(
                weight,
                crate::TensorFormat {
                    precision: crate::Precision::F16,
                    layout: crate::Layout::row_sharded(4),
                },
            );
        let program = crate::planner::plan(
            &graph,
            &crate::planner::boundary_layouts(&graph, &config),
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
        let mut timeline = Timeline::default();
        mid::analyze_observed(&program, &BTreeMap::new(), &mut timeline).unwrap();
        let parameter = program
            .inputs
            .iter()
            .find(|input| input.kind == crate::GraphInputKind::Parameter)
            .unwrap()
            .value;
        let root = timeline
            .values
            .iter()
            .find(|v| v.id == parameter.index())
            .unwrap()
            .root;
        assert!(timeline.steps.len() >= 2);
        assert!(timeline.steps.iter().all(|step| step.live.contains(&root)));
    }
}
