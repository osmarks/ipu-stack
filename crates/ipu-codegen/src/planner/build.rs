//! Deterministic whole-device lowering with explicit, canonical boundaries.

use crate::PipelineConfig;
use crate::estimate::{CostModel, MemoryPeaks};
use crate::graph::{
    ComputeGraph, GraphInputKind, Operation, OperationId, OperationKind, Repeat, ValueId,
};
use crate::mid::{
    CoordinateMapping, MidInput, MidOperation, MidOperationKind, MidProgram, MidRegion, MidRepeat,
    MidValueId,
};
use crate::planner::bind::{ValueBuilder, canonical, emit_selected, ensure_format, flat, lookup};
use crate::planner::cache::FragmentCache;
use crate::planner::candidates::{CandidateSearch, OutputDemands};
use crate::planner::catalogue::{
    candidate_active_tile_counts, default_operator_candidates, operator_candidates_for_tile_count,
    shape_aware_active_tile_counts,
};
use crate::planner::error::{LoweringError, LoweringResult};
use crate::planner::operator::{OperandMaterialization, OperatorDispatch, OperatorPlan};
use crate::planner::parameter_homes;
use crate::planner::recipe::{Candidate, Recipe};
use crate::tensor::{TensorFormat, TensorShape, TensorType};
use std::collections::BTreeMap;

pub(crate) fn build_candidate(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    fragments: &FragmentCache,
    recipe: &Recipe,
) -> LoweringResult<Candidate> {
    let mut selected = select(graph, config, costs, fragments, recipe)?;
    let options = &selected.recipe.options;
    let mut program = selected.program;
    if options.cast_before_copies {
        program.reorder_casts();
    }
    program.compose_copies();
    if !config.diagnostic_checkpoints {
        program = program
            .with_elementwise_fusions(
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            )
            .unwrap_or(program);
        if options.packing_rows != 0 {
            program = program
                .with_distributed_packing(options.packing_rows)
                .unwrap_or(program);
        }
        program.group_reductions(options.parallel_reductions)?;
        if options.disjoint_copy_sources {
            program.distribute_preparation(config.diagnostic_checkpoints)?;
        }
        if options.reuse_cast_inputs {
            program.reuse_cast_inputs();
        }
    }
    if !options.tile_mapping.is_empty() {
        program.remap_tiles(&options.tile_mapping)?;
    }
    selected.program = program;
    selected
        .program
        .refresh_estimates()
        .ok_or(LoweringError::InvalidImplementation)?;
    Ok(selected)
}

pub(crate) fn select(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    fragments: &FragmentCache,
    recipe: &Recipe,
) -> LoweringResult<Candidate> {
    if config.tile_count == 0 {
        return Err(LoweringError::EmptyTileGroup);
    }
    let expanded_config = (config.shape_aware_active_tile_counts
        && config.operator_candidates == default_operator_candidates(config.tile_count))
    .then(|| {
        let mut expanded = config.clone();
        let existing_counts = candidate_active_tile_counts(config.tile_count);
        expanded.operator_candidates.extend(
            shape_aware_active_tile_counts(config.tile_count, graph.value_shapes().values())
                .into_iter()
                .filter(|tiles| !existing_counts.contains(tiles))
                .flat_map(operator_candidates_for_tile_count),
        );
        expanded
    });
    let config = expanded_config.as_ref().unwrap_or(config);
    let mut builder = Builder {
        graph,
        config,
        costs,
        fragments,
        recipe: recipe.clone(),
        state: ValueBuilder::default(),
        values: BTreeMap::new(),
        alternatives: BTreeMap::new(),
        optimizing: !recipe.plans.is_empty() || !recipe.open_boundaries.is_empty(),
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
                    canonical(
                        &input.shape,
                        precision,
                        config.tile_count,
                        config.capacity_baseline,
                    )
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
    parameter_homes::assign_parameter_tiles(
        &mut builder.state.values,
        &inputs
            .iter()
            .filter(|input| input.kind == GraphInputKind::Parameter)
            .map(|input| input.value)
            .collect::<Vec<_>>(),
        &BTreeMap::new(),
        config.tile_count,
    )?;
    let mut program = MidProgram {
        tile_count: config.tile_count,
        inputs,
        operations,
        outputs,
        values: builder.state.values,
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
        peak_memory: MemoryPeaks::default(),
    };
    crate::mid::ownership::bind_owners(&mut program.operations, &mut program.values)?;
    program.validate()?;
    Ok(Candidate {
        program,
        recipe: builder.recipe,
        alternatives: builder.alternatives,
    })
}

struct Builder<'a, C> {
    graph: &'a ComputeGraph,
    config: &'a PipelineConfig,
    costs: &'a C,
    fragments: &'a FragmentCache,
    recipe: Recipe,
    state: ValueBuilder,
    values: BTreeMap<ValueId, MidValueId>,
    alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
    optimizing: bool,
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
            .flat_map(|op| self.graph.operation_inputs(op))
            .chain(required.iter().copied())
        {
            *uses.entry(input).or_insert(0) += 1;
        }
        let demands = OutputDemands::new(source, shapes, self.config);
        for (index, operation) in source.iter().enumerate() {
            if let OperationKind::Repeat(repeat) = &operation.kind {
                self.repeat(operation, repeat, &uses, &mut operations)?;
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
            let search = CandidateSearch::new(
                operation,
                &source[index + 1..],
                required.contains(&operation.results[0]),
                &uses,
                self.config,
                &demands,
            );
            let mut plans = if let Some(plan) = self.recipe.plans.get(&operation.id) {
                vec![plan.clone()]
            } else {
                search.generate(
                    &types,
                    &parameters,
                    &automatic,
                    shape,
                    self.costs,
                    self.fragments,
                )?
            };
            // Prefer the AMP attention family over the scalar reference kernel
            // when this shape has a supported whole-device implementation.
            if plans
                .iter()
                .any(|plan| matches!(plan.dispatch, OperatorDispatch::Attention { .. }))
            {
                plans.retain(|plan| matches!(plan.dispatch, OperatorDispatch::Attention { .. }));
            }
            let rank = |cycles, memory: MemoryPeaks| {
                let (first, last) = if self.optimizing {
                    (cycles, memory.total)
                } else {
                    (memory.total, cycles)
                };
                (
                    first,
                    if self.config.capacity_baseline {
                        memory.maximum_standard_allocation
                    } else {
                        0
                    },
                    memory.exchange_rows,
                    last,
                )
            };
            let selected = if let Some(plan) = self.recipe.plans.get(&operation.id) {
                plan.clone()
            } else {
                plans
                    .iter()
                    .filter_map(|plan| {
                        if !self.config.capacity_baseline {
                            let (inputs, output) = plan.tensor_types(&types, shape);
                            let implementation = self.fragments.get(plan, &inputs, &output)?;
                            let mut state = ValueBuilder::default();
                            let mut conversions = Vec::new();
                            for ((source, requirement), &automatic) in
                                types.iter().zip(&plan.inputs).zip(&automatic)
                            {
                                let id = state.value(operation.results[0], source.clone());
                                if automatic {
                                    state.automatic_inputs.insert(id);
                                }
                                ensure_format(
                                    id,
                                    requirement.format.clone(),
                                    // This baseline prices whole-input transitions even
                                    // for panel candidates. Their bounded lifetime is
                                    // represented by the family fragment's memory peak.
                                    OperandMaterialization::Complete,
                                    operation.id,
                                    self.costs,
                                    &mut state,
                                    &mut conversions,
                                );
                            }
                            let memory = implementation.peak_memory;
                            let cycles = self
                                .costs
                                .operator_cycle_override(plan, &inputs, &output)
                                .unwrap_or(implementation.estimated_cycles)
                                .saturating_add(state.conversion_cycles);
                            return Some((rank(cycles, memory), plan));
                        }
                        // Lower the actual boundary -> operator -> boundary sequence.
                        // The operator alone omits live source buffers and cast/pack
                        // temporaries, which can be larger than its own scratch.
                        let mut state = ValueBuilder::default();
                        let mut initial = Vec::new();
                        for (index, (&origin, source)) in
                            operation.inputs.iter().zip(&types).enumerate()
                        {
                            let id = state.value(origin, source.clone());
                            if automatic[index] {
                                state.automatic_inputs.insert(id);
                            }
                            if parameters[index] {
                                state.parameter_values.insert(id);
                            }
                            if let Some(&copies) = self.state.copies.get(&ids[index]) {
                                state.copies.insert(id, copies);
                            }
                            initial.push(id);
                        }
                        let mut sequence = Vec::new();
                        let results = emit_selected(
                            operation,
                            &initial,
                            shape.clone(),
                            plan,
                            self.recipe.open_boundaries.contains(&operation.results[0]),
                            self.config,
                            self.costs,
                            self.fragments,
                            &mut state,
                            &mut sequence,
                        )
                        .ok()?;
                        // Inputs with later consumers remain live through conversion.
                        let mut live = results;
                        live.extend(
                            initial
                                .iter()
                                .zip(&operation.inputs)
                                .filter(|(_, origin)| uses.get(origin) != Some(&1))
                                .map(|(&id, _)| id),
                        );
                        let fragment = crate::estimate::region_program(
                            self.config.tile_count,
                            &initial,
                            &sequence,
                            &live,
                            &state.values,
                        );
                        let mut fragment = fragment;
                        if self.recipe.options.cast_before_copies {
                            fragment.reorder_casts();
                        }
                        fragment.compose_copies();
                        let (cycles, memory) = crate::estimate::analyze_with_budget(
                            &fragment,
                            &BTreeMap::new(),
                            self.config,
                        )?;
                        let cycles = cycles.total;

                        Some((rank(cycles, memory), plan))
                    })
                    .min_by_key(|(score, _)| *score)
                    .map(|(_, plan)| plan.clone())
                    .ok_or(LoweringError::NoCandidate(operation.id))?
            };
            if !self.recipe.plans.contains_key(&operation.id) {
                self.alternatives.insert(operation.id, plans);
            }
            self.recipe.plans.insert(operation.id, selected.clone());
            let results = emit_selected(
                operation,
                &ids,
                shape.clone(),
                &selected,
                self.recipe.open_boundaries.contains(&operation.results[0]),
                self.config,
                self.costs,
                self.fragments,
                &mut self.state,
                &mut operations,
            )?;
            self.values
                .extend(operation.results.iter().copied().zip(results));
        }
        Ok(operations)
    }

    fn repeat(
        &mut self,
        operation: &Operation,
        repeat: &Repeat,
        uses: &BTreeMap<ValueId, usize>,
        operations: &mut Vec<MidOperation>,
    ) -> LoweringResult<()> {
        let mut inputs = operation
            .inputs
            .iter()
            .map(|id| lookup(&self.values, *id))
            .collect::<LoweringResult<Vec<_>>>()?;
        // Repeat updates carried storage in place. Parameters and values used
        // elsewhere can seed that state, but cannot donate their allocation.
        for (index, input) in inputs.iter_mut().take(repeat.carried_inputs).enumerate() {
            if self.state.parameter_values.contains(input)
                || uses.get(&operation.inputs[index]) != Some(&1)
            {
                let source = self.state.get(*input).clone();
                let result = self.state.value(source.origin, source.tensor_type);
                self.state.values[result.index() as usize].owners = source.owners.clone();
                operations.push(MidOperation {
                    source: Some(operation.id),
                    inputs: vec![*input],
                    results: vec![result],
                    kind: MidOperationKind::Copy {
                        policy: crate::CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Automatic,
                        mapping: CoordinateMapping::default(),
                        reuse_local: false,
                    },
                });
                *input = result;
            }
        }
        let sequences = repeat
            .iterated_inputs
            .iter()
            .map(|id| {
                self.graph.sequences()[id.index() as usize]
                    .values
                    .iter()
                    .take(repeat.count as usize)
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
                self.state.copies.insert(id, repeat.count);
            }
            self.values.insert(origin, id);
            arguments.push(id);
        }
        let mut body = self.region(
            &repeat.body.operations,
            &repeat.body.yields,
            &repeat.body.value_shapes,
        )?;
        // A body may select a compact native home for an invariant parameter.
        // Bind the outer allocation to that same format, just as for sequences.
        for (index, input) in inputs.iter_mut().enumerate().skip(repeat.carried_inputs) {
            let target = self.state.get(arguments[index]).tensor_type.format.clone();
            *input = ensure_format(
                *input,
                target,
                OperandMaterialization::Complete,
                operation.id,
                self.costs,
                &mut self.state,
                operations,
            );
        }
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
                operation.id,
                self.costs,
                &mut self.state,
                &mut body,
            ));
        }
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
                },
            }),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::estimate::{Ipu21CostModel, MemoizedCostModel};
    use crate::tensor::Precision;

    use super::*;

    #[test]
    fn repeated_invariant_parameters_keep_their_selected_home() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [16, 64]).unwrap();
        let weight = graph.parameter("weight", [64, 64]).unwrap();
        let y = graph
            .repeat(3, [x], [weight], [], |body, args| {
                Ok(vec![body.gemm(args.carried[0], args.invariants[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([y]).unwrap();
        let config = PipelineConfig::new(8)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(weight, Precision::F16);
        let baseline = build_candidate(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
            &Recipe::baseline(&config),
        )
        .unwrap();
        let op = baseline
            .program
            .operations
            .iter()
            .find(|op| matches!(op.kind, MidOperationKind::Repeat(_)))
            .unwrap();
        let MidOperationKind::Repeat(repeat) = &op.kind else {
            unreachable!()
        };
        let home = &baseline.program.values[op.inputs[1].index() as usize];
        let argument = &baseline.program.values[repeat.body.arguments[1].index() as usize];
        assert_eq!(home.tensor_type, argument.tensor_type);
        assert_eq!(home.owners, argument.owners);
        let low = crate::low::expand::expand_tiles(&baseline.program, false).unwrap();
        crate::place(&crate::lower_to_tiles(&low, false)).unwrap();
    }

    #[test]
    fn canonical_activation_boundaries_do_not_pad_tail_rows() {
        for precision in [Precision::F16, Precision::F8F143 { scale_exponent: -4 }] {
            for shape in [[1, 729, 1152], [2, 729, 3456], [8, 729, 4304]] {
                let shape = TensorShape(shape.to_vec());
                let layout = canonical(&shape, precision, 1472, true);
                assert_eq!(layout.tiling.replicas, 1);
                assert_eq!(layout.tiling.tile_count, 729);
                assert_eq!(layout.padded_shape(&shape).unwrap(), shape);
                let resolved = layout.resolve(&shape).unwrap();
                assert!(!resolved.has_empty_shards());
                for (_, extents) in resolved.shard_extents().unwrap() {
                    assert!(
                        extents
                            .iter()
                            .all(|axis| axis.logical_end == axis.physical_end)
                    );
                }
            }
        }
    }

    #[test]
    fn baseline_has_canonical_boundaries_and_compact_repeated_weights() {
        for count in [1, 3] {
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("x", [16, 64]).unwrap();
            let mut config = PipelineConfig::new(8).with_automatic_input(input, Precision::F16);
            config.capacity_baseline = count == 3;
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
            let costs = MemoizedCostModel::new(&Ipu21CostModel);
            let baseline = build_candidate(
                &graph,
                &config,
                &costs,
                &crate::planner::cache::FragmentCache::default(),
                &Recipe::baseline(&config),
            )
            .unwrap();
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
                run.call().unwrap();
            }
            crate::place(&low).unwrap();
        }
    }
}
