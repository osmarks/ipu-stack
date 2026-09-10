//! Reusable region searches, independent of enclosing value numbering and prefixes.
//!
//! A search owns one immutable body/configuration context. Boundary keys describe
//! everything visible to that body. It returns a frontier, not a selected plan:
//! enclosing liveness and memory are checked after attachment by the outer beam.

use super::*;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Argument {
    origin: ValueId,
    tensor_type: TensorType,
    tile_offset: u16,
    storage_class: MidValueId,
    automatic: bool,
    parameter: bool,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct RegionBoundary(Vec<Argument>);

impl RegionBoundary {
    pub(super) fn new(
        origins: &[ValueId],
        bindings: &[MidValueId],
        state: &LoweringState,
        automatic: impl Fn(usize) -> bool,
        parameter: impl Fn(usize) -> bool,
    ) -> Self {
        assert_eq!(origins.len(), bindings.len());
        let mut groups = BTreeMap::new();
        Self(
            origins
                .iter()
                .zip(bindings)
                .enumerate()
                .map(|(index, (&origin, &id))| {
                    let value = state.get(id);
                    Argument {
                        origin,
                        tensor_type: value.tensor_type.clone(),
                        tile_offset: value.tile_offset,
                        storage_class: *groups
                            .entry(value.storage_group)
                            .or_insert(MidValueId(index as u32)),
                        automatic: automatic(index),
                        parameter: parameter(index),
                    }
                })
                .collect(),
        )
    }

    fn compact_parameters(
        &self,
        copies: &BTreeMap<ValueId, u32>,
        config: &PipelineConfig,
    ) -> Option<Self> {
        let mut boundary = self.clone();
        let mut changed = false;
        for argument in &mut boundary.0 {
            if argument.automatic
                && argument.parameter
                && copies.get(&argument.origin).copied().unwrap_or(1) > 1
            {
                if let Some(layout) = super::ownership::compact_parameter_layout(
                    &argument.tensor_type,
                    copies[&argument.origin],
                    config,
                ) {
                    argument.tensor_type.format.layout = layout;
                }
                argument.automatic = false;
                changed = true;
            }
        }
        changed.then_some(boundary)
    }

    fn state(&self) -> (LoweringState, BTreeMap<ValueId, MidValueId>) {
        let mut state = LoweringState::default();
        let mut values = BTreeMap::new();
        for argument in &self.0 {
            let id = state.value_in_storage_group(
                argument.origin,
                argument.tensor_type.clone(),
                argument.storage_class,
            );
            state.values[id.index() as usize].tile_offset = argument.tile_offset;
            if argument.automatic {
                state.automatic_inputs.insert(id);
            }
            if argument.parameter {
                state.parameter_values.insert(id);
            }
            values.insert(argument.origin, id);
        }
        (state, values)
    }
}

pub(super) struct RegionSearch<'a, C> {
    source: &'a [Operation],
    outputs: &'a [ValueId],
    shapes: &'a BTreeMap<ValueId, TensorShape>,
    graph: &'a ComputeGraph,
    config: &'a PipelineConfig,
    costs: &'a C,
    cache: BTreeMap<
        (RegionBoundary, RegionPlanningConstraints),
        LoweringResult<Arc<Vec<RegionCandidate>>>,
    >,
    searches: usize,
    hits: usize,
}

impl<'a, C: CostModel> RegionSearch<'a, C> {
    pub(super) fn new(
        source: &'a [Operation],
        outputs: &'a [ValueId],
        shapes: &'a BTreeMap<ValueId, TensorShape>,
        graph: &'a ComputeGraph,
        config: &'a PipelineConfig,
        costs: &'a C,
    ) -> Self {
        Self {
            source,
            outputs,
            shapes,
            graph,
            config,
            costs,
            cache: BTreeMap::new(),
            searches: 0,
            hits: 0,
        }
    }

    pub(super) fn plan(
        &mut self,
        boundary: RegionBoundary,
        constraints: RegionPlanningConstraints,
    ) -> LoweringResult<Arc<Vec<RegionCandidate>>> {
        let key = (boundary, constraints);
        if let Some(candidates) = self.cache.get(&key) {
            self.hits += 1;
            return candidates.clone();
        }
        self.searches += 1;
        let started = std::time::Instant::now();
        tracing::info!(
            context = self.searches,
            cache_hits = self.hits,
            operations = self.source.len(),
            "searching region body context"
        );
        let (mut state, mut values) = key.0.state();
        let argument_count = state.values.len();
        let mut candidates = lower_operation_candidates(
            self.source,
            self.outputs,
            &mut values,
            self.shapes,
            self.graph,
            self.config,
            self.costs,
            &mut state,
            &key.1,
        )
        .map(|branches| {
            Arc::new(
                branches
                    .into_iter()
                    .map(|branch| RegionCandidate {
                        branch,
                        argument_count,
                    })
                    .collect::<Vec<_>>(),
            )
        });
        tracing::info!(
            context = self.searches,
            elapsed_ms = started.elapsed().as_millis() as u64,
            candidates = candidates.as_ref().map_or(0, |plans| plans.len()),
            "searched region body context"
        );
        if matches!(
            candidates,
            Err(LoweringError::NoCandidate(_) | LoweringError::InsufficientMemory { .. })
        ) && let Some(boundary) = key
            .0
            .compact_parameters(&key.1.allocation_copies, self.config)
        {
            tracing::info!("retrying region with compact persistent parameter homes");
            candidates = self.plan(boundary, key.1.clone());
        }
        self.cache.insert(key, candidates.clone());
        candidates
    }
}

impl<C> Drop for RegionSearch<'_, C> {
    fn drop(&mut self) {
        tracing::info!(
            searches = self.searches,
            cache_hits = self.hits,
            operations = self.source.len(),
            "region body planning"
        );
    }
}

pub(super) struct RegionCandidate {
    branch: BeamBranch,
    argument_count: usize,
}

impl RegionCandidate {
    pub(super) fn attach(
        &self,
        bindings: &[MidValueId],
        state: &mut LoweringState,
    ) -> (
        Vec<MidValueId>,
        BTreeMap<ValueId, MidValueId>,
        Vec<MidOperation>,
    ) {
        assert_eq!(bindings.len(), self.argument_count);
        let base = state.values.len() as u32;
        let remap = |id: MidValueId| MidValueId(base + id.index());
        for value in &self.branch.state.values {
            let mut value = value.clone();
            value.id = remap(value.id);
            value.storage_group = if (value.storage_group.index() as usize) < bindings.len() {
                state
                    .get(bindings[value.storage_group.index() as usize])
                    .storage_group
            } else {
                remap(value.storage_group)
            };
            state.values.push(value);
        }
        state.automatic_inputs.extend(
            self.branch
                .state
                .automatic_inputs
                .iter()
                .copied()
                .map(remap),
        );
        state.parameter_values.extend(
            self.branch
                .state
                .parameter_values
                .iter()
                .copied()
                .map(remap),
        );
        let mut operations = self.branch.operations.clone();
        remap_operations(&mut operations, &remap);
        (
            (0..self.argument_count)
                .map(|index| remap(MidValueId(index as u32)))
                .collect(),
            self.branch
                .values
                .iter()
                .map(|(&origin, &id)| (origin, remap(id)))
                .collect(),
            operations,
        )
    }
}

fn remap_operations(operations: &mut [MidOperation], remap: &impl Fn(MidValueId) -> MidValueId) {
    for operation in operations {
        for id in operation.inputs.iter_mut().chain(&mut operation.results) {
            *id = remap(*id);
        }
        match &mut operation.kind {
            MidOperationKind::Operator {
                deferred_inputs, ..
            } => {
                // Implementations have their own local namespace, not region IDs.
                for input in deferred_inputs.iter_mut().flatten() {
                    input.producer = remap(input.producer);
                    input.source = remap(input.source);
                }
            }
            MidOperationKind::Repeat(repeat) => {
                for id in repeat
                    .iterated_inputs
                    .iter_mut()
                    .flatten()
                    .chain(&mut repeat.body.arguments)
                    .chain(&mut repeat.body.yields)
                {
                    *id = remap(*id);
                }
                remap_operations(&mut repeat.body.operations, remap);
            }
            MidOperationKind::Primitive(_) | MidOperationKind::Convert(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_boundary_only_freezes_repeated_automatic_parameters() {
        let mut graph = ComputeGraph::new();
        let origins: Vec<_> = (0..4)
            .map(|i| graph.host_input(format!("x{i}"), [1152]).unwrap())
            .collect();
        let mut state = LoweringState::default();
        let bindings: Vec<_> = origins
            .iter()
            .map(|&origin| {
                state.value(
                    origin,
                    TensorType {
                        shape: graph.value_shapes()[&origin].clone(),
                        format: TensorFormat {
                            precision: Precision::F16,
                            layout: Layout::logical_linear(288, 4),
                        },
                    },
                )
            })
            .collect();
        let boundary = RegionBoundary::new(&origins, &bindings, &state, |i| i != 3, |i| i != 1);
        let copies = BTreeMap::from([(origins[0], 27), (origins[1], 27), (origins[3], 27)]);
        let config = PipelineConfig::new(1472);
        let compact = boundary.compact_parameters(&copies, &config).unwrap();
        assert!(!compact.0[0].automatic);
        assert_eq!(compact.0[0].tensor_type.format.layout.tiling.tile_count, 9);
        assert!(compact.0[1..] == boundary.0[1..]);
        assert!(compact.compact_parameters(&copies, &config).is_none());
    }

    #[test]
    fn boundary_ignores_numbering_but_preserves_planning_constraints() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [1, 8, 16]).unwrap();
        let y = graph.host_input("y", [1, 8, 16]).unwrap();
        let ty = TensorType {
            shape: graph.value_shapes()[&x].clone(),
            format: TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(4),
            },
        };
        let mut a = LoweringState::default();
        let ids = [a.value(x, ty.clone()), a.value(y, ty.clone())];
        let boundary = RegionBoundary::new(&[x, y], &ids, &a, |_| true, |_| false);
        let mut b = LoweringState::default();
        b.value(x, ty.clone()); // unrelated prefix
        let other = [b.value(x, ty.clone()), b.value(y, ty)];
        let equivalent = RegionBoundary::new(&[x, y], &other, &b, |_| true, |_| false);
        assert!(boundary == equivalent);
        assert!(boundary != RegionBoundary::new(&[x, y], &ids, &a, |_| false, |_| false));
        assert!(boundary != RegionBoundary::new(&[x, y], &ids, &a, |_| true, |_| true));
        b.values[other[1].index() as usize].storage_group = other[0];
        assert!(boundary != RegionBoundary::new(&[x, y], &other, &b, |_| true, |_| false));
        b.values[other[1].index() as usize].storage_group = other[1];
        b.values[other[0].index() as usize].tile_offset = 1;
        assert!(boundary != RegionBoundary::new(&[x, y], &other, &b, |_| true, |_| false));
        b.values[other[0].index() as usize].tile_offset = 0;
        b.values[other[0].index() as usize]
            .tensor_type
            .format
            .layout = Layout::row_sharded(2);
        assert!(boundary != RegionBoundary::new(&[x, y], &other, &b, |_| true, |_| false));

        let config = PipelineConfig::new(4);
        let outputs = [x, y];
        let mut search = RegionSearch::new(
            &[],
            &outputs,
            graph.value_shapes(),
            &graph,
            &config,
            &Ipu21CostModel,
        );
        let first = search
            .plan(boundary.clone(), RegionPlanningConstraints::default())
            .unwrap();
        let cached = search
            .plan(equivalent, RegionPlanningConstraints::default())
            .unwrap();
        assert!(Arc::ptr_eq(&first, &cached));
        search
            .plan(
                boundary.clone(),
                RegionPlanningConstraints {
                    allocation_copies: BTreeMap::from([(x, 2)]),
                    ..Default::default()
                },
            )
            .unwrap();
        search
            .plan(
                boundary,
                RegionPlanningConstraints {
                    required_equal_formats: vec![(x, y)],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!((search.searches, search.hits), (3, 1));
        let prefix = b.values.clone();
        let (arguments, _, _) = cached[0].attach(&other, &mut b);
        assert_eq!(&b.values[..prefix.len()], &prefix);
        for (&argument, &binding) in arguments.iter().zip(&other) {
            assert_eq!(b.get(argument).storage_group, b.get(binding).storage_group);
        }
    }

    #[test]
    fn region_search_retains_body_alternatives_and_relocates_them() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [1, 16, 32]).unwrap();
        let w = graph.parameter("w", [1, 32, 32]).unwrap();
        let y = graph.gemm(x, w).unwrap();
        graph.set_outputs([y]).unwrap();
        let mut state = LoweringState::default();
        let bindings = [x, w].map(|origin| {
            state.value(
                origin,
                TensorType {
                    shape: graph.value_shapes()[&origin].clone(),
                    format: TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::row_sharded(16),
                    },
                },
            )
        });
        let boundary = RegionBoundary::new(&[x, w], &bindings, &state, |_| true, |i| i == 1);
        let config = PipelineConfig::new(16);
        let mut search = RegionSearch::new(
            graph.operations(),
            graph.outputs(),
            graph.value_shapes(),
            &graph,
            &config,
            &Ipu21CostModel,
        );
        let plans = search
            .plan(boundary.clone(), RegionPlanningConstraints::default())
            .unwrap();
        assert!(plans.len() > 1, "body alternatives were discarded");
        let impossible = RegionPlanningConstraints {
            allocation_copies: BTreeMap::from([(w, u32::MAX)]),
            ..Default::default()
        };
        let first_error = search
            .plan(boundary.clone(), impossible.clone())
            .err()
            .unwrap();
        let cached_error = search.plan(boundary, impossible).err().unwrap();
        assert_eq!(first_error, cached_error);
        // The impossible context also tries compact homes, then caches failure.
        assert_eq!((search.searches, search.hits), (3, 1));

        for plan in plans.iter() {
            let mut attached = state.clone();
            let (arguments, values, operations) = plan.attach(&bindings, &mut attached);
            assert_eq!(arguments.len(), 2);
            for operation in &operations {
                for id in operation.read_values().chain(&operation.results) {
                    assert!((id.index() as usize) >= state.values.len());
                    assert_eq!(attached.get(*id).id, *id);
                }
            }
            let program = MidProgram {
                tile_count: 16,
                values: attached.values,
                inputs: arguments
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| MidInput {
                        name: format!("arg{i}"),
                        kind: if i == 1 {
                            GraphInputKind::Parameter
                        } else {
                            GraphInputKind::Host
                        },
                        value,
                    })
                    .collect(),
                operations,
                outputs: vec![values[&y]],
                ..Default::default()
            };
            super::super::expand_tiles(&program).unwrap();
        }
    }
}
