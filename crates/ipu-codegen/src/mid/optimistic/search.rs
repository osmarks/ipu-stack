use super::*;
type ConversionCache = std::collections::HashMap<(TensorType, TensorType), Vec<ConversionPath>>;

#[derive(Clone, Default)]
struct State {
    graph: DiagnosticMidGraph,
    live: BTreeMap<ValueId, usize>,
}
impl State {
    fn value(&mut self, origin: ValueId, tensor: TensorType) -> usize {
        let id = self.graph.values.len();
        self.graph.values.push(Value { origin, tensor });
        id
    }
    fn convert(
        &self,
        id: usize,
        target: &TensorType,
        tiles: u16,
        options: &SearchOptions,
        cache: &mut ConversionCache,
    ) -> Result<Vec<(Self, usize)>, SearchError> {
        let origin = self.graph.values[id].origin;
        if let Some(existing) = self
            .graph
            .values
            .iter()
            .position(|v| v.origin == origin && &v.tensor == target)
        {
            return Ok(vec![(self.clone(), existing)]);
        }
        let mut result = Vec::new();
        let key = (self.graph.values[id].tensor.clone(), target.clone());
        if !cache.contains_key(&key) {
            cache.insert(
                key.clone(),
                enumerate_conversions(&key.0, &key.1, tiles, options)?,
            );
        }
        for path in cache[&key].clone() {
            let mut state = self.clone();
            let mut at = id;
            for transform in path.steps {
                if let Some(existing) = state
                    .graph
                    .values
                    .iter()
                    .position(|v| v.origin == origin && v.tensor == transform.to)
                {
                    at = existing;
                    continue;
                }
                let output = state.value(origin, transform.to.clone());
                state.graph.steps.push(Step {
                    sources: vec![],
                    inputs: vec![at],
                    outputs: vec![output],
                    cycles: transform.cycles,
                    assumptions: transform.assumptions.clone(),
                    kind: StepKind::Transform(transform),
                });
                at = output;
            }
            refresh(&mut state.graph);
            result.push((state, at));
        }
        Ok(result)
    }
}

/// Plan an entire small, straight-line high graph. Input formats come from the
/// supplied config; output formats are explicit to make comparisons meaningful.
pub fn plan_graph(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    outputs: BTreeMap<ValueId, TensorFormat>,
    options: &SearchOptions,
) -> Result<SearchReport, SearchError> {
    let mut inputs = BTreeMap::new();
    for input in graph.inputs().iter().filter(|input| {
        graph
            .operations()
            .iter()
            .any(|op| op.inputs.contains(&input.value))
    }) {
        let format = config
            .inputs
            .get(&input.value)
            .cloned()
            .or_else(|| {
                config
                    .automatic_inputs
                    .get(&input.value)
                    .map(|&precision| TensorFormat {
                        precision,
                        layout: Layout::row_sharded(config.tile_count),
                    })
            })
            .ok_or_else(|| invalid(format!("missing boundary input {:?}", input.value)))?;
        inputs.insert(input.value, format);
    }
    plan_region(
        graph,
        &RegionRequest {
            operations: 0..graph.operations().len(),
            inputs,
            outputs,
        },
        config,
        options,
    )
}

/// Search a contiguous top-level high-graph region, preserving every result
/// which escapes it. No selected mid plan is required. Repeat is an explicit
/// boundary for this entry point; it is never silently unrolled or omitted.
pub fn plan_region(
    graph: &ComputeGraph,
    request: &RegionRequest,
    config: &PipelineConfig,
    options: &SearchOptions,
) -> Result<SearchReport, SearchError> {
    let ops = graph
        .operations()
        .get(request.operations.clone())
        .ok_or_else(|| invalid("invalid operation range"))?;
    if ops.is_empty()
        || ops.len() > options.max_operations
        || options.beam_width == 0
        || options.max_expansions == 0
        || options.conversion_frontier == 0
        || options.max_conversion_steps == 0
        || config.tile_count == 0
    {
        return Err(invalid("empty region or invalid/exceeded search limits"));
    }
    let mut needed = BTreeSet::new();
    let mut defined = BTreeSet::new();
    for op in ops {
        if matches!(op.kind, OperationKind::Repeat(_)) {
            return Err(SearchError::UnsupportedOperation(op.id));
        }
        if op.results.len() != 1 {
            return Err(SearchError::UnsupportedOperation(op.id));
        }
        for input in &op.inputs {
            if !defined.contains(input) {
                needed.insert(*input);
            }
        }
        defined.extend(op.results.iter().copied());
    }
    if needed != request.inputs.keys().copied().collect() {
        return Err(invalid(
            "boundary inputs must exactly cover region free values",
        ));
    }
    let mut escaped = graph.operations()[request.operations.end..]
        .iter()
        .flat_map(|op| op.inputs.iter())
        .chain(graph.outputs())
        .filter(|v| defined.contains(v))
        .copied()
        .collect::<BTreeSet<_>>();
    for op in &graph.operations()[request.operations.end..] {
        if let OperationKind::Repeat(repeat) = &op.kind {
            for sequence in &repeat.iterated_inputs {
                if let Some(sequence) = graph.sequences().iter().find(|s| s.id == *sequence) {
                    escaped.extend(
                        sequence
                            .values
                            .iter()
                            .filter(|v| defined.contains(v))
                            .copied(),
                    );
                }
            }
        }
    }
    if !escaped.is_subset(&request.outputs.keys().copied().collect())
        || request.outputs.is_empty()
        || request.outputs.keys().any(|v| !defined.contains(v))
    {
        return Err(invalid(
            "outputs must include every escaping regional result",
        ));
    }
    let mut initial = State::default();
    initial.graph.tile_count = config.tile_count;
    for (&origin, format) in &request.inputs {
        let tensor = TensorType {
            shape: graph
                .value_shape(origin)
                .ok_or_else(|| invalid("unknown input"))?
                .clone(),
            format: format.clone(),
        };
        if !valid_tensor(&tensor, config.tile_count) {
            return Err(invalid("invalid input format"));
        }
        let id = initial.value(origin, tensor);
        initial.graph.inputs.push(id);
        initial.live.insert(origin, id);
    }
    for (&origin, format) in &request.outputs {
        let tensor = TensorType {
            shape: graph
                .value_shape(origin)
                .ok_or_else(|| invalid("unknown output"))?
                .clone(),
            format: format.clone(),
        };
        if !valid_tensor(&tensor, config.tile_count) {
            return Err(invalid("invalid output format"));
        }
    }
    let parameters = graph
        .inputs()
        .iter()
        .filter(|i| i.kind == GraphInputKind::Parameter)
        .map(|i| i.value)
        .collect::<BTreeSet<_>>();
    let demands = OutputDemands::new(ops, graph.value_shapes(), config);
    let costs = crate::estimate::MemoizedCostModel::new(&Ipu21CostModel, config.tile_count);
    let mut conversion_cache = ConversionCache::new();
    let mut beam = vec![initial];
    let mut report = SearchReport::default();
    for (index, op) in ops.iter().enumerate() {
        let operation_budget =
            report.expanded + (options.max_expansions - report.expanded) / (ops.len() - index);
        let mut next = Vec::new();
        let shape = graph
            .value_shape(op.results[0])
            .ok_or_else(|| invalid("unknown result"))?;
        for (state_index, state) in beam.iter().enumerate() {
            let state_budget =
                report.expanded + (operation_budget - report.expanded) / (beam.len() - state_index);
            let inputs = op
                .inputs
                .iter()
                .map(|v| state.graph.values[state.live[v]].tensor.clone())
                .collect::<Vec<_>>();
            let parameter_inputs = op
                .inputs
                .iter()
                .map(|v| parameters.contains(v))
                .collect::<Vec<_>>();
            // Reuse unpruned production candidate construction and backward
            // consumer demands. In particular, do not apply its conversion
            // availability filter or assume the selected producer ownership.
            let candidates = plans(
                op,
                &inputs,
                &parameter_inputs,
                shape,
                config,
                &costs,
                true,
                None,
                &[],
                demands.get(op.results[0]),
            );
            for plan in candidates {
                if report.expanded >= state_budget {
                    report.truncated = true;
                    break;
                }
                report.expanded += 1;
                let (required, output) = plan.tensor_types(&inputs, shape);
                if !required.iter().all(|t| valid_tensor(t, config.tile_count))
                    || !valid_tensor(&output, config.tile_count)
                {
                    continue;
                }
                let Some(implementation) = costs.implementation(&plan, &required, &output) else {
                    report.rejected_implementations += 1;
                    continue;
                };
                let price = implementation.estimated_cycles;
                let mut branches = vec![(state.clone(), Vec::new())];
                for (&origin, target) in op.inputs.iter().zip(&required) {
                    let mut converted = Vec::new();
                    for (branch, ids) in branches {
                        for (child, id) in branch.convert(
                            branch.live[&origin],
                            target,
                            config.tile_count,
                            options,
                            &mut conversion_cache,
                        )? {
                            let mut ids = ids.clone();
                            ids.push(id);
                            converted.push((child, ids));
                        }
                    }
                    converted.sort_by_key(|(s, _)| score(&s.graph));
                    if converted.len() > options.beam_width {
                        report.truncated = true;
                    }
                    let reference = converted
                        .iter()
                        .find(|(s, _)| s.graph.assumptions.is_empty())
                        .cloned();
                    converted.truncate(options.beam_width);
                    if let Some(reference) = reference
                        && !converted
                            .iter()
                            .any(|(s, _)| s.graph.assumptions.is_empty())
                    {
                        converted.push(reference);
                    }
                    branches = converted;
                }
                for (mut child, ids) in branches {
                    let result = child.value(op.results[0], output.clone());
                    child.live.insert(op.results[0], result);
                    child.graph.steps.push(Step {
                        sources: vec![op.id],
                        inputs: ids,
                        outputs: vec![result],
                        kind: StepKind::Algorithm {
                            plan: plan.clone(),
                            implementation: implementation.clone(),
                        },
                        cycles: CycleEstimate {
                            optimistic: price,
                            conservative: price,
                        },
                        assumptions: BTreeSet::new(),
                    });
                    let remaining = ops[index + 1..]
                        .iter()
                        .flat_map(|op| op.inputs.iter())
                        .chain(request.outputs.keys())
                        .copied()
                        .collect::<BTreeSet<_>>();
                    child.live.retain(|v, _| remaining.contains(v));
                    refresh(&mut child.graph);
                    if !fits(&child.graph, config) {
                        report.rejected_memory += 1;
                        continue;
                    }
                    next.push(child);
                    if next.len() > options.beam_width.saturating_mul(4) {
                        next = prune(next, options.beam_width, &mut report);
                    }
                }
            }
        }
        beam = prune(next, options.beam_width, &mut report);
        if beam.is_empty() {
            return Err(SearchError::NoCandidates {
                expanded: report.expanded,
                memory: report.rejected_memory,
                implementations: report.rejected_implementations,
            });
        }
    }
    for (&origin, format) in &request.outputs {
        let mut next = Vec::new();
        for state in beam {
            let target = TensorType {
                shape: graph.value_shape(origin).unwrap().clone(),
                format: format.clone(),
            };
            let converted = state.convert(
                state.live[&origin],
                &target,
                config.tile_count,
                options,
                &mut conversion_cache,
            )?;
            for (mut child, id) in converted {
                child.graph.outputs.push(id);
                refresh(&mut child.graph);
                if !fits(&child.graph, config) {
                    report.rejected_memory += 1;
                    continue;
                }
                next.push(child);
            }
        }
        beam = prune(next, options.beam_width, &mut report);
    }
    for state in beam {
        if let Some(fused) = super::fusion::fuse(&state.graph, ops, options.max_fusion_operations) {
            report.candidates.push(fused);
        }
        if !report.candidates.contains(&state.graph) {
            report.candidates.push(state.graph);
        }
    }
    if report.candidates.is_empty() {
        return Err(SearchError::NoCandidates {
            expanded: report.expanded,
            memory: report.rejected_memory,
            implementations: report.rejected_implementations,
        });
    }
    report.candidates.sort_by_key(score);
    Ok(report)
}

fn score(graph: &DiagnosticMidGraph) -> (u64, u64, u64, usize) {
    let cycles = graph
        .steps
        .iter()
        .fold(CycleEstimate::default(), |sum, step| sum.plus(step.cycles));
    (
        cycles.optimistic,
        cycles.conservative,
        graph.memory.total,
        graph.assumptions.len(),
    )
}
fn prune(mut states: Vec<State>, limit: usize, report: &mut SearchReport) -> Vec<State> {
    states.sort_by_key(|s| score(&s.graph));
    let baseline = states
        .iter()
        .find(|s| s.graph.assumptions.is_empty())
        .cloned();
    let signature = |s: &State| {
        s.live
            .iter()
            .map(|(&v, &id)| (v, s.graph.values[id].tensor.clone()))
            .collect::<Vec<_>>()
    };
    let mut result: Vec<State> = Vec::new();
    let mut remainder = Vec::new();
    let mut unique = Vec::<State>::new();
    let mut buckets = std::collections::HashMap::<u64, Vec<usize>>::new();
    for state in states {
        let key = fingerprint(&state.graph);
        let bucket = buckets.entry(key).or_default();
        if bucket
            .iter()
            .any(|&i| unique[i].graph == state.graph && unique[i].live == state.live)
        {
            continue;
        }
        bucket.push(unique.len());
        unique.push(state);
    }
    for state in unique {
        let sig = signature(&state);
        // Preserve different live output formats before filling the beam with
        // cost variants of a single distribution. Materialization caches differ,
        // so equal signatures alone are NOT enough for dominance elimination.
        if result.iter().any(|s| signature(s) == sig) {
            remainder.push(state);
        } else {
            result.push(state);
        }
    }
    result.extend(remainder);
    if result.len() > limit {
        report.truncated = true;
    }
    result.truncate(limit);
    if let Some(baseline) = baseline
        && !result.iter().any(|s| s.graph.assumptions.is_empty())
    {
        result.push(baseline);
    }
    result
}

pub(super) fn refresh(graph: &mut DiagnosticMidGraph) {
    graph.cycles = CycleEstimate::default();
    graph.assumptions.clear();
    graph.memory = MemoryPeaks::default();
    graph.minimum_live_bytes_per_tile = 0;
    let n = graph.values.len();
    let mut first = vec![usize::MAX; n];
    let mut last = vec![0; n];
    for &id in &graph.inputs {
        first[id] = 0;
    }
    for (index, step) in graph.steps.iter().enumerate() {
        for &id in &step.outputs {
            first[id] = index;
            last[id] = index;
        }
        for &id in &step.inputs {
            last[id] = last[id].max(index);
        }
        graph.cycles = graph.cycles.plus(step.cycles);
        graph.assumptions.extend(step.assumptions.iter().cloned());
    }
    for &id in &graph.outputs {
        last[id] = graph.steps.len();
    }
    let mut roots = (0..n).collect::<Vec<_>>();
    for step in &graph.steps {
        if matches!(
            &step.kind,
            StepKind::Transform(Transform {
                kind: TransformKind::Alias,
                ..
            })
        ) {
            let root = roots[step.inputs[0]];
            roots[step.outputs[0]] = root;
            first[root] = first[root].min(first[step.outputs[0]]);
            last[root] = last[root].max(last[step.outputs[0]]);
        }
    }
    let physical = graph
        .values
        .iter()
        .map(|v| {
            v.tensor
                .format
                .layout
                .resolve(&v.tensor.shape)
                .map_or(u64::MAX, |r| {
                    r.physical_elements()
                        .saturating_mul(v.tensor.format.precision.bytes())
                })
        })
        .collect::<Vec<_>>();
    for index in 0..=graph.steps.len() {
        let mut usage = MemoryUsage::default();
        let mut allocation = 0;
        let mut device_bytes = 0u64;
        for (id, value) in graph.values.iter().enumerate() {
            if roots[id] == id && first[id] <= index && last[id] >= index {
                device_bytes = device_bytes.saturating_add(physical[id]);
                let bytes = crate::estimate::maximum_shard_bytes(&value.tensor);
                usage.add_class(value.tensor.format.layout.memory_class, bytes);
                if value.tensor.format.layout.memory_class == MemoryClass::Ipu21Standard {
                    allocation = allocation.max(bytes);
                }
            }
        }
        if let Some(Step {
            kind: StepKind::Algorithm { implementation, .. },
            ..
        }) = graph.steps.get(index)
        {
            // Includes some endpoint storage twice: retain this conservative
            // scratch bound rather than inventing physical ownership alignment.
            usage.standard = usage
                .standard
                .saturating_add(implementation.peak_memory.standard);
            usage.interleaved = usage
                .interleaved
                .saturating_add(implementation.peak_memory.interleaved);
        }
        graph.memory.observe(usage, allocation);
        graph.minimum_live_bytes_per_tile = graph
            .minimum_live_bytes_per_tile
            .max(device_bytes.div_ceil(u64::from(graph.tile_count.max(1))));
    }
}

fn fingerprint(graph: &DiagnosticMidGraph) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for value in &graph.values {
        value.origin.hash(&mut h);
        value.tensor.hash(&mut h);
    }
    for step in &graph.steps {
        step.inputs.hash(&mut h);
        step.outputs.hash(&mut h);
        step.sources.hash(&mut h);
        match &step.kind {
            StepKind::Algorithm { plan, .. } => plan.hash(&mut h),
            StepKind::Transform(t) => {
                t.from.hash(&mut h);
                t.to.hash(&mut h);
            }
            StepKind::FusedElementwise { operations } => {
                for op in operations {
                    op.id.hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

fn fits(graph: &DiagnosticMidGraph, config: &PipelineConfig) -> bool {
    // Reject proven capacity violations, not the pessimistic sum of unrelated
    // owners' maxima. Neither test establishes aligned placement feasibility.
    graph
        .minimum_live_bytes_per_tile
        .saturating_add(config.standard_memory_reservation_bytes)
        <= config.tile_memory_budget_bytes
        && graph.values.iter().all(|v| {
            crate::estimate::maximum_shard_bytes(&v.tensor) <= config.tile_memory_budget_bytes
        })
}
