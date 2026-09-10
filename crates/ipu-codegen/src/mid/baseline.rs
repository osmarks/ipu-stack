//! Deterministic whole-device lowering with explicit, canonical boundaries.
use super::*;

#[derive(Clone, Default)]
pub(crate) struct Recipe {
    pub plans: BTreeMap<OperationId, OperatorPlan>,
    pub open_boundaries: BTreeSet<ValueId>,
    pub late_casts: BTreeSet<OperationId>,
}

pub(crate) struct Baseline {
    pub program: MidProgram,
    pub recipe: Recipe,
    pub alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
}

pub(crate) fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    recipe: &Recipe,
) -> LoweringResult<Baseline> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let mut builder = Builder {
        graph,
        config,
        costs,
        recipe: recipe.clone(),
        state: LoweringState::default(),
        values: BTreeMap::new(),
        alternatives: BTreeMap::new(),
        copies: BTreeMap::new(),
    };
    let mut inputs = Vec::new();
    for input in graph.inputs() {
        let fixed = config.inputs.get(&input.value);
        let precision = fixed
            .map(|f| f.precision)
            .or_else(|| config.automatic_inputs.get(&input.value).copied())
            .ok_or(LoweringError::MissingInputType(input.value))?;
        let tensor = TensorType {
            shape: input.shape.clone(),
            format: fixed.cloned().unwrap_or_else(|| TensorFormat {
                precision,
                layout: if input.kind == GraphInputKind::Parameter {
                    flat(&input.shape, precision, config.tile_count)
                } else {
                    canonical(&input.shape, precision, config.tile_count)
                },
            }),
        };
        let id = builder.state.value(input.value, tensor);
        if input.kind == GraphInputKind::Parameter {
            builder.state.parameter_values.insert(id);
            if fixed.is_none() {
                builder.state.automatic_inputs.insert(id);
            }
        }
        builder.values.insert(input.value, id);
        inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value: id,
        });
    }
    let operations = builder.region(graph.operations(), graph.outputs(), graph.value_shapes())?;
    let outputs = graph
        .outputs()
        .iter()
        .map(|id| lookup(&builder.values, *id))
        .collect::<LoweringResult<Vec<_>>>()?;
    let program = MidProgram {
        tile_count: config.tile_count,
        inputs,
        operations,
        outputs,
        values: builder.state.values,
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
        peak_memory: MemoryPeaks::default(),
    };
    let mut program =
        implementation::resolve(program).ok_or(LoweringError::InvalidImplementation)?;
    let (cycles, peak) = crate::estimate::analyze_mid(&program, &BTreeMap::new())
        .ok_or(LoweringError::InvalidImplementation)?;
    program.estimated_cycles = cycles.total;
    program.estimated_exchange_cycles = cycles.exchange;
    program.peak_memory = peak;
    Ok(Baseline {
        program,
        recipe: builder.recipe,
        alternatives: builder.alternatives,
    })
}

fn canonical(shape: &TensorShape, precision: Precision, tiles: u16) -> Layout {
    if shape.0.len() >= 2 {
        Layout::row_sharded(tiles)
    } else {
        flat(shape, precision, tiles)
    }
}

fn flat(shape: &TensorShape, precision: Precision, tiles: u16) -> Layout {
    let mut grain = (8 / precision.bytes()) as u32;
    while !shape.elements().is_multiple_of(u64::from(grain)) {
        grain /= 2;
    }
    let owners = (shape.elements() / u64::from(grain)).min(u64::from(tiles)) as u16;
    Layout::logical_linear(owners, grain)
}

struct Builder<'a, C> {
    graph: &'a ComputeGraph,
    config: &'a PipelineConfig,
    costs: &'a C,
    recipe: Recipe,
    state: LoweringState,
    values: BTreeMap<ValueId, MidValueId>,
    alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
    copies: BTreeMap<MidValueId, u32>,
}

impl<C: CostModel> Builder<'_, C> {
    fn region(
        &mut self,
        source: &[Operation],
        required: &[ValueId],
        shapes: &BTreeMap<ValueId, TensorShape>,
    ) -> LoweringResult<Vec<MidOperation>> {
        let mut operations = Vec::new();
        let mut uses = BTreeMap::new();
        for input in source
            .iter()
            .flat_map(|op| operation_graph_inputs(op, self.graph))
            .chain(required.iter().copied())
        {
            *uses.entry(input).or_insert(0) += 1;
        }
        let demands = OutputDemands::new(source, shapes, self.config);
        for (index, operation) in source.iter().enumerate() {
            if let OperationKind::Repeat(repeat) = &operation.kind {
                self.repeat(operation, repeat, &mut operations)?;
                continue;
            }
            let ids = operation
                .inputs
                .iter()
                .map(|id| lookup(&self.values, *id))
                .collect::<LoweringResult<Vec<_>>>()?;
            let types = ids
                .iter()
                .map(|id| self.state.get(*id).tensor_type.clone())
                .collect::<Vec<_>>();
            let parameters = ids
                .iter()
                .map(|id| self.state.parameter_values.contains(id))
                .collect::<Vec<_>>();
            let automatic = ids
                .iter()
                .map(|id| self.state.automatic_inputs.contains(id))
                .collect::<Vec<_>>();
            let shape = shapes
                .get(&operation.results[0])
                .ok_or(LoweringError::MissingShape(operation.results[0]))?;
            let mut search = CandidateSearch::new(
                operation,
                &source[index + 1..],
                required.contains(&operation.results[0]),
                &uses,
                self.config,
                &demands,
            );
            let plans = search.generate(&types, &parameters, &automatic, shape, self.costs)?;
            let selected = if let Some(plan) = self.recipe.plans.get(&operation.id) {
                plan.clone()
            } else {
                plans
                    .iter()
                    .filter_map(|plan| {
                        let (inputs, output) = plan.tensor_types(&types, shape);
                        let implementation = self.costs.implementation(plan, &inputs, &output)?;
                        let memory = implementation.peak_memory;
                        Some((
                            (
                                memory.total,
                                memory.exchange_rows,
                                implementation.estimated_cycles,
                            ),
                            plan,
                        ))
                    })
                    .min_by_key(|(score, _)| *score)
                    .map(|(_, plan)| plan.clone())
                    .ok_or(LoweringError::NoCandidate(operation.id))?
            };
            self.alternatives.insert(operation.id, plans);
            self.recipe.plans.insert(operation.id, selected.clone());
            // Persistent storage is chosen independently of compute replication.
            // Only automatic homes can change; explicitly bound inputs stay fixed.
            for (&id, requirement) in ids.iter().zip(&selected.requirements.inputs) {
                if self.state.automatic_inputs.contains(&id)
                    && self.state.parameter_values.contains(&id)
                {
                    let mut tensor = self.state.get(id).tensor_type.clone();
                    tensor.format.layout.order = requirement.format.layout.order;
                    let copies = self.copies.get(&id).copied().unwrap_or(1);
                    let layout = ownership::compact_parameter_layout(&tensor, copies, self.config)
                        .ok_or(LoweringError::NoCandidate(operation.id))?;
                    self.state.retarget_automatic_input(id, layout);
                }
            }
            let parameter_origins = ids
                .iter()
                .filter(|id| self.state.parameter_values.contains(id))
                .map(|id| self.state.get(*id).origin)
                .collect::<BTreeSet<_>>();
            let previous_values = self.state.values.len();
            apply_selected_plan(
                operation,
                shape.clone(),
                selected,
                &vec![!self.recipe.late_casts.contains(&operation.id); ids.len()],
                &vec![false; ids.len()],
                self.costs,
                &mut self.values,
                &mut self.state,
                &mut operations,
            );
            // Materialized parameter copies are ordinary temporaries, not members
            // of the persistent sequence's ownership/replication group.
            for value in &mut self.state.values[previous_values..] {
                if parameter_origins.contains(&value.origin) {
                    value.storage_group = value.id;
                    value.tile_offset = 0;
                }
            }
            let output = operation.results[0];
            if !self.recipe.open_boundaries.contains(&output) {
                let id = self.values[&output];
                let value = self.state.get(id);
                let target = TensorFormat {
                    precision: value.tensor_type.format.precision,
                    layout: canonical(
                        shape,
                        value.tensor_type.format.precision,
                        self.config.tile_count,
                    ),
                };
                let id = ensure_format(
                    id,
                    target,
                    OperandMaterialization::Complete,
                    false,
                    operation.id,
                    self.costs,
                    &mut self.state,
                    &mut operations,
                );
                self.values.insert(output, id);
            }
        }
        restore_unclaimed_deferred_costs(&mut operations);
        Ok(operations)
    }

    fn repeat(
        &mut self,
        operation: &Operation,
        repeat: &Repeat,
        operations: &mut Vec<MidOperation>,
    ) -> LoweringResult<()> {
        let inputs = operation
            .inputs
            .iter()
            .map(|id| lookup(&self.values, *id))
            .collect::<LoweringResult<Vec<_>>>()?;
        let sequences = repeat
            .iterated_inputs
            .iter()
            .map(|id| {
                self.graph.sequences()[id.index() as usize]
                    .values
                    .iter()
                    .map(|id| lookup(&self.values, *id))
                    .collect::<LoweringResult<Vec<_>>>()
            })
            .collect::<LoweringResult<Vec<_>>>()?;
        for sequence in &sequences {
            let group = self.state.get(sequence[0]).storage_group;
            for &id in sequence {
                self.state.values[id.index() as usize].storage_group = group;
            }
        }
        let bindings = inputs
            .iter()
            .copied()
            .chain(sequences.iter().map(|s| s[0]))
            .collect::<Vec<_>>();
        let mut arguments = Vec::new();
        for (index, (&origin, &binding)) in repeat.body.arguments.iter().zip(&bindings).enumerate()
        {
            let id = self
                .state
                .derived_value(binding, self.state.get(binding).tensor_type.clone());
            self.state.values[id.index() as usize].origin = origin;
            if self.state.automatic_inputs.contains(&binding) {
                self.state.automatic_inputs.insert(id);
            }
            if index >= inputs.len() {
                self.copies.insert(id, repeat.count);
            }
            self.values.insert(origin, id);
            arguments.push(id);
        }
        let mut body = self.region(
            &repeat.body.operations,
            &repeat.body.yields,
            &repeat.body.value_shapes,
        )?;
        let mut iterated_inputs = Vec::new();
        for (index, sequence) in sequences.iter().enumerate() {
            let target = self
                .state
                .get(arguments[inputs.len() + index])
                .tensor_type
                .format
                .clone();
            let converted = sequence
                .iter()
                .map(|&id| {
                    ensure_format(
                        id,
                        target.clone(),
                        OperandMaterialization::Complete,
                        false,
                        operation.id,
                        self.costs,
                        &mut self.state,
                        operations,
                    )
                })
                .collect();
            iterated_inputs.push(converted);
        }
        let mut yields = Vec::new();
        for (index, &origin) in repeat.body.yields.iter().enumerate() {
            let value = lookup(&self.values, origin)?;
            yields.push(ensure_format(
                value,
                self.state.get(inputs[index]).tensor_type.format.clone(),
                OperandMaterialization::Complete,
                false,
                operation.id,
                self.costs,
                &mut self.state,
                &mut body,
            ));
        }
        let peak = region_peak_memory_with_multiplicity(
            self.config,
            &arguments,
            &body,
            &yields,
            &self.state.values,
            &self.copies,
        );
        let cycles = body.iter().map(|op| op.estimated_cycles).sum::<u64>();
        let exchange = body
            .iter()
            .map(|op| op.estimated_exchange_cycles)
            .sum::<u64>();
        let results = operation
            .results
            .iter()
            .zip(&inputs)
            .map(|(&origin, &input)| {
                let id = self
                    .state
                    .derived_value(input, self.state.get(input).tensor_type.clone());
                self.state.values[id.index() as usize].origin = origin;
                self.values.insert(origin, id);
                id
            })
            .collect();
        operations.push(MidOperation {
            source: Some(operation.id),
            inputs,
            results,
            kind: MidOperationKind::Repeat(MidRepeat {
                count: repeat.count,
                carried_inputs: repeat.carried_inputs,
                invariant_inputs: repeat.invariant_inputs,
                iterated_inputs,
                body: MidRegion {
                    arguments,
                    operations: body,
                    yields,
                    estimated_cycles: cycles,
                    peak_memory: peak,
                },
            }),
            estimated_cycles: cycles.saturating_mul(u64::from(repeat.count)),
            estimated_exchange_cycles: exchange.saturating_mul(u64::from(repeat.count)),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_has_canonical_boundaries_and_compact_repeated_weights() {
        for count in [1, 3] {
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("x", [16, 64]).unwrap();
            let mut config = PipelineConfig::new(8).with_automatic_input(input, Precision::F16);
            let weights = (0..count)
                .map(|i| {
                    let id = graph.parameter(format!("w{i}"), [64, 64]).unwrap();
                    config = config.clone().with_automatic_input(id, Precision::F16);
                    id
                })
                .collect::<Vec<_>>();
            let sequence = graph.value_sequence("weights", weights).unwrap();
            let output = graph
                .repeat(count, [input], [], [sequence], |body, args| {
                    let x = body.gemm(args.carried[0], args.iterated[0])?;
                    Ok(vec![body.gelu(x)?])
                })
                .unwrap()[0];
            graph.set_outputs([output]).unwrap();
            let costs = MemoizedCostModel::new(&Ipu21CostModel, config.tile_count);
            let baseline = lower(&graph, &config, &costs, &Recipe::default()).unwrap();
            let repeat = baseline
                .program
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    MidOperationKind::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .unwrap();
            assert_eq!(repeat.count, count);
            assert_eq!(repeat.iterated_inputs[0].len(), count as usize);
            let argument = &baseline.program.values[repeat.body.arguments[1].index() as usize];
            assert_eq!(argument.tensor_type.format.layout.tiling.replicas, 1);
            let carried = &baseline.program.values[repeat.body.arguments[0].index() as usize];
            let yielded = &baseline.program.values[repeat.body.yields[0].index() as usize];
            assert_eq!(carried.tensor_type.format, yielded.tensor_type.format);
            let low = crate::low::expand::expand_tiles(&baseline.program, false).unwrap();
            let low = crate::lower_to_tiles(&low, false);
            for run in &low.kernel_runs {
                crate::validate_kernel_run(run).unwrap();
            }
            crate::place(&low).unwrap();
        }
    }
}
