//! Opt-in explanations of the planner's existing memory estimate.
use super::*;
use crate::{ComputeGraph, MidOperationKind, PipelineConfig, Primitive};
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
            MidOperationKind::Primitive(Primitive::Compute { kernel, .. }) => format!("{kernel:?}"),
            MidOperationKind::Primitive(primitive) => format!("{primitive:?}"),
            MidOperationKind::Convert(plan) => format!("Convert {:?}", plan.strategy),
            MidOperationKind::Repeat(repeat) => format!("Repeat {} boundary", repeat.count),
            MidOperationKind::Operator { .. } => "Operator".into(),
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

fn names(graph: &ComputeGraph) -> BTreeMap<u32, (String, String)> {
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
        graph: &ComputeGraph,
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
    gemm_output_packing: String,
    attention_products: String,
    timeline: Timeline,
}

fn profile(
    scope: &str,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    copies: &BTreeMap<MidValueId, u32>,
) -> Option<Profile> {
    let program = mid::resolved_region(config.tile_count, initial, operations, outputs, values)?;
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
            .min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)),
        interleaved_budget_bytes: u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES),
        standard_contiguous_overflow_bytes: peak.standard_contiguous_overflow_with_reservation(
            config.standard_memory_reservation_bytes,
        ),
        fits_budget: peak.fits_ipu21_with_budget(
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
        gemm_output_packing: format!("{:?}", config.gemm_output_packing),
        attention_products: format!("{:?}", config.attention_products),
        timeline,
    })
}

/// Called only for exhausted memory shortlists and selected planner finalists.
/// Diagnostics never force every candidate to retain a detailed timeline.
pub(crate) fn write(
    scope: &str,
    graph: &ComputeGraph,
    config: &PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    copies: &BTreeMap<MidValueId, u32>,
) -> crate::LoweringResult<()> {
    let Some(directory) = &config.memory_profile_directory else {
        return Ok(());
    };
    let profile = profile(
        scope, graph, config, initial, operations, outputs, values, copies,
    )
    .ok_or(crate::LoweringError::InvalidImplementation)?;
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
    .map_err(|error| crate::LoweringError::MemoryProfile(error.to_string()))?;
    tracing::info!(path = %result.display(), scope,
        peak_step = profile.total_peak_step, effective_peak_bytes = profile.effective_peak_bytes,
        "wrote planner memory profile");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeline_reconstructs_estimates_with_repeat_aliases_and_padding() {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("state</script>", [1, 8, 32]).unwrap();
        let weights = (0..3)
            .map(|i| graph.parameter(format!("w{i}"), [1, 32, 32]).unwrap())
            .collect::<Vec<_>>();
        let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
        let output = graph
            .repeat(3, [input], [], [sequence], |body, args| {
                Ok(vec![body.gemm(args.carried[0], args.iterated[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(8).with_automatic_input(input, Precision::F16);
        for weight in weights {
            config = config.with_automatic_input(weight, Precision::F16);
        }
        let program = crate::mid::lower(&graph, &config, &Ipu21CostModel).unwrap();
        let initial = program
            .inputs
            .iter()
            .map(|input| input.value)
            .collect::<Vec<_>>();
        let report = profile(
            "test",
            &graph,
            &config,
            &initial,
            &program.operations,
            &program.outputs,
            &program.values,
            &BTreeMap::new(),
        )
        .unwrap();
        let (_, expected) = mid::region_estimate(
            &config,
            &initial,
            &program.operations,
            &program.outputs,
            &program.values,
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(report.peak, expected);
        let mut allocations = BTreeMap::new();
        for value in &report.timeline.values {
            assert_eq!(value.bytes, value.aligned_bytes * u64::from(value.copies));
            let entry = allocations.entry(value.root).or_insert((0, &value.class));
            entry.0 = entry.0.max(value.bytes);
            entry.1 = &value.class;
        }
        assert!(
            allocations.len() < report.timeline.values.len(),
            "Repeat aliases were lost"
        );
        assert!(
            report
                .timeline
                .values
                .iter()
                .any(|v| v.aligned_bytes > v.shard_bytes)
        );
        assert!(report.timeline.steps.iter().any(|s| s.execution_count == 3));
        for step in &report.timeline.steps {
            for tile in 0..usize::from(config.tile_count) {
                let mut usage = step.tile_scratch[tile];
                for id in &step.live {
                    let aliases = report
                        .timeline
                        .values
                        .iter()
                        .filter(|v| v.root == *id)
                        .collect::<Vec<_>>();
                    let bytes = aliases.iter().map(|v| v.tile_bytes[tile]).max().unwrap();
                    let class = usize::from(aliases[0].class == "Ipu21Interleaved");
                    usage[class] += bytes;
                }
                assert_eq!(
                    usage, step.tile_usage[tile],
                    "step {}, tile {tile}",
                    step.index
                );
                assert!(usage[0] + usage[1] <= step.coarse_usage.total());
            }
        }
        let peak = &report.timeline.steps[report.total_peak_step.unwrap()];
        assert_eq!(peak.usage.total(), report.peak.total);
        assert!(report.timeline.values.iter().any(|v| v.group == "weights"));

        let repeat = program
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                MidOperationKind::Repeat(repeat) => Some((op, repeat)),
                _ => None,
            })
            .unwrap();
        let copies = repeat
            .1
            .body
            .arguments
            .iter()
            .skip(repeat.0.inputs.len())
            .copied()
            .map(|id| (id, 3))
            .collect();
        let body = profile(
            "body",
            &graph,
            &config,
            &repeat.1.body.arguments,
            &repeat.1.body.operations,
            &repeat.1.body.yields,
            &program.values,
            &copies,
        )
        .unwrap();
        assert!(body.timeline.values.iter().any(|v| v.copies == 3));
        assert!(
            body.timeline
                .steps
                .iter()
                .all(|step| step.execution_count == 1)
        );

        let directory = std::env::temp_dir().join(format!(
            "ipu-memory-profile-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        config.memory_profile_directory = Some(directory.clone());
        write(
            "test",
            &graph,
            &config,
            &initial,
            &program.operations,
            &program.outputs,
            &program.values,
            &BTreeMap::new(),
        )
        .unwrap();
        for file in std::fs::read_dir(&directory).unwrap() {
            let path = file.unwrap().path();
            let text = std::fs::read_to_string(&path).unwrap();
            if path.extension().unwrap() == "json" {
                let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(parsed["peak"]["total"], report.peak.total);
            } else {
                assert!(!text.contains("state</script>"));
                assert!(text.contains("state\\u003c/script>"));
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
