//! Deterministic whole-device lowering with explicit, canonical boundaries.

use crate::compile::PipelineConfig;
use crate::estimate::{CostModel, MemoryPeaks};
use crate::graph::{
    ComputeGraph, GraphInputKind, Operation, OperationId, OperationKind, Repeat, ValueId,
};
use crate::mid::{
    CoordinateMapping, LocalSite, MidInput, MidOperation, MidOperationKind, MidProgram, MidRegion,
    MidRepeat, MidValueId,
};
use crate::planner::bind::{ValueBuilder, emit_selected, ensure_format, lookup};
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
use crate::tensor::{
    AxisTiling, Layout, Padding, Precision, TensorAxis, TensorFormat, TensorShape, TensorTiling,
    TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn build_candidate(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
    fragments: &FragmentCache,
    recipe: &Recipe,
) -> LoweringResult<Candidate> {
    let mut selected = select(graph, config, costs, fragments, recipe)?;
    selected.recipe.normalize(config);
    let mut program = selected.program;
    selected.cast_sites = program.reorder_casts(&BTreeSet::new());
    selected.recipe.resolve_cast_choices(&selected.cast_sites)?;
    if !selected.recipe.cast_before_copies.is_empty() {
        program.reorder_casts(&selected.recipe.cast_before_copies);
    }
    program.compose_copies();
    selected.program = if config.diagnostic_checkpoints {
        program
    } else {
        program
            .with_elementwise_fusions(
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            )
            .unwrap_or(program)
    };
    if !config.diagnostic_checkpoints {
        selected.packing_choices = selected
            .program
            .packing_choices(&super::proposals::PACKING_ROWS, &selected.recipe.packing);
        selected.program.apply_packing(&selected.recipe.packing)?;
        selected.grouping_choices = (2..=config.max_parallel_reductions)
            .map(|limit| selected.program.propose_reduction_groups(limit))
            .filter(|proposal| !proposal.reductions.is_empty())
            .collect();
    } else {
        if !selected.recipe.packing.is_empty() {
            return Err(crate::mid::ProgramError::Invalid(
                "packing choices cannot apply with diagnostic checkpoints".into(),
            )
            .into());
        }
    }
    if config.diagnostic_checkpoints && !selected.recipe.reduction_groups.is_empty() {
        return Err(crate::mid::ProgramError::Invalid(
            "reduction grouping cannot apply with diagnostic checkpoints".into(),
        )
        .into());
    }
    selected
        .program
        .group_reductions(&selected.recipe.reduction_groups)?;
    let homes = selected
        .program
        .propose_preparation_homes(config.diagnostic_checkpoints);
    if !homes.is_empty() {
        selected.grouping_choices.push(crate::mid::GroupProposal {
            reductions: selected.recipe.reduction_groups.clone(),
            homes,
        });
    }
    let storage = selected.recipe.cast_storage.as_ref().unwrap();
    if config.diagnostic_checkpoints {
        if !storage.sites.is_empty() || !storage.operators.is_empty() {
            return Err(crate::mid::ProgramError::Invalid(
                "cast-storage overrides cannot apply with diagnostic checkpoints".into(),
            )
            .into());
        }
    } else {
        selected.cast_storage_sites = selected.program.reuse_cast_inputs(storage);
        if let Some(site) = storage
            .sites
            .keys()
            .find(|site| !selected.cast_storage_sites.contains(site))
        {
            return Err(LoweringError::UnavailableCastStorageChoice(site.clone()));
        }
    }
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
    // Defaults may name an operator whose selected implementation emits no
    // work. Validate provenance against the graph, not the surviving mid sites.
    let sources = graph
        .walk_operations()
        .map(|op| op.id)
        .collect::<BTreeSet<_>>();
    let requests = recipe
        .owners
        .operators
        .keys()
        .map(|source| ("ownership", source))
        .chain(
            recipe
                .cast_storage
                .iter()
                .flat_map(|policy| policy.operators.keys())
                .map(|source| ("cast-storage", source)),
        );
    if let Some((policy, source)) = requests
        .into_iter()
        .find(|(_, source)| !sources.contains(source))
    {
        return Err(crate::mid::ProgramError::Invalid(format!(
            "{policy} policy names unknown operator {source:?}"
        ))
        .into());
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
        copies: BTreeMap::new(),
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
    program.apply_ownership(&builder.recipe.owners)?;
    program.validate()?;
    Ok(Candidate {
        cast_sites: BTreeSet::new(),
        cast_storage_sites: BTreeSet::new(),
        packing_choices: BTreeMap::new(),
        grouping_choices: Vec::new(),
        program,
        recipe: builder.recipe,
        alternatives: builder.alternatives,
    })
}

fn canonical(shape: &TensorShape, precision: Precision, tiles: u16, capacity: bool) -> Layout {
    if shape.0.len() < 2 {
        return flat(shape, precision, tiles);
    }
    // Keep whole rows local, but use every available row owner. Coarse
    // element-sized boundaries concentrate residuals and conversion buffers on
    // the same small tile group, even when most of the device is free.
    let rows = shape.0[shape.0.len() - 2];
    let owners = if capacity {
        rows.min(u32::from(tiles)) as u16
    } else {
        let row_bytes = shape.elements() / u64::from(rows) * precision.bytes();
        let capacity =
            (u64::from(ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE) / row_bytes).max(1);
        let block = (1u32 << capacity.min(u64::from(u32::MAX)).ilog2())
            .min(rows)
            .max(rows.div_ceil(u32::from(tiles)));
        rows.div_ceil(block) as u16
    };
    Layout::row_major(TensorTiling {
        tile_count: owners,
        replicas: 1,
        axes: vec![AxisTiling::new(
            TensorAxis::FromEnd(2),
            owners,
            1,
            Padding::Reject,
        )],
    })
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
    fragments: &'a FragmentCache,
    recipe: Recipe,
    state: ValueBuilder,
    values: BTreeMap<ValueId, MidValueId>,
    alternatives: BTreeMap<OperationId, Vec<OperatorPlan>>,
    copies: BTreeMap<MidValueId, u32>,
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
            let (selected, early_cast) = if let Some(plan) = self.recipe.plans.get(&operation.id) {
                (
                    plan.clone(),
                    self.recipe.early_casts.contains(&operation.id),
                )
            } else {
                plans
                    .iter()
                    .flat_map(|plan| [(plan, false), (plan, true)])
                    .filter(|(plan, early)| {
                        !early
                            || self.config.capacity_baseline
                                && types.iter().zip(&plan.requirements.inputs).any(
                                    |(input, requirement)| {
                                        input.format.precision != requirement.format.precision
                                    },
                                )
                    })
                    .filter_map(|(plan, early_cast)| {
                        let early_cast =
                            early_cast || self.recipe.early_casts.contains(&operation.id);
                        if !self.config.capacity_baseline {
                            let (inputs, output) = plan.tensor_types(&types, shape);
                            let implementation = self.fragments.get(plan, &inputs, &output)?;
                            let mut state = ValueBuilder::default();
                            let mut conversions = Vec::new();
                            for (index, ((source, requirement), &automatic)) in types
                                .iter()
                                .zip(&plan.requirements.inputs)
                                .zip(&automatic)
                                .enumerate()
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
                                    LocalSite::from("input").at(index as u32),
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
                            return Some((rank(cycles, memory), (plan, early_cast)));
                        }
                        // Lower the actual boundary -> operator -> boundary sequence.
                        // The operator alone omits live source buffers and cast/pack
                        // temporaries, which can be larger than its own scratch.
                        let mut state = ValueBuilder::default();
                        let mut values = BTreeMap::new();
                        let mut initial = Vec::new();
                        for (index, (&origin, source)) in
                            operation.inputs.iter().zip(&types).enumerate()
                        {
                            let mut source = source.clone();
                            if automatic[index] && parameters[index] {
                                source.format.layout.order =
                                    plan.requirements.inputs[index].format.layout.order;
                                source.format.layout = parameter_homes::compact_parameter_layout(
                                    &source,
                                    self.copies.get(&ids[index]).copied().unwrap_or(1),
                                    self.config,
                                )?;
                            }
                            let id = state.value(origin, source);
                            if automatic[index] && !parameters[index] {
                                state.automatic_inputs.insert(id);
                            }
                            initial.push(id);
                            values.insert(origin, id);
                        }
                        let mut sequence = Vec::new();
                        emit_selected(
                            operation,
                            shape.clone(),
                            plan,
                            self.costs,
                            self.fragments,
                            self.config.tile_count,
                            &mut values,
                            &mut state,
                            &mut sequence,
                        )
                        .ok()?;
                        let mut result = values[&operation.results[0]];
                        if !self.recipe.open_boundaries.contains(&operation.results[0]) {
                            let precision = state.get(result).tensor_type.format.precision;
                            result = ensure_format(
                                result,
                                TensorFormat {
                                    precision,
                                    layout: canonical(
                                        shape,
                                        precision,
                                        self.config.tile_count,
                                        self.config.capacity_baseline,
                                    ),
                                },
                                OperandMaterialization::Complete,
                                operation.id,
                                "boundary",
                                self.costs,
                                &mut state,
                                &mut sequence,
                            );
                        }
                        // Inputs with later consumers remain live through conversion.
                        let mut live = vec![result];
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
                        if early_cast {
                            let sites = fragment.reorder_casts(&BTreeSet::new());
                            fragment.reorder_casts(&sites);
                        }
                        fragment.compose_copies();
                        let (cycles, memory) = crate::estimate::analyze_with_budget(
                            &fragment,
                            &BTreeMap::new(),
                            self.config,
                        )?;
                        let cycles = cycles.total;

                        Some((rank(cycles, memory), (plan, early_cast)))
                    })
                    .min_by_key(|(score, _)| *score)
                    .map(|(_, (plan, early_cast))| (plan.clone(), early_cast))
                    .ok_or(LoweringError::NoCandidate(operation.id))?
            };
            if early_cast {
                self.recipe.early_casts.insert(operation.id);
            }
            if !self.recipe.plans.contains_key(&operation.id) {
                self.alternatives.insert(operation.id, plans);
            }
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
                    let layout =
                        parameter_homes::compact_parameter_layout(&tensor, copies, self.config)
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
            emit_selected(
                operation,
                shape.clone(),
                &selected,
                self.costs,
                self.fragments,
                self.config.tile_count,
                &mut self.values,
                &mut self.state,
                &mut operations,
            )?;
            // Materialized parameter copies are ordinary temporaries, not members
            // of the persistent sequence's ownership/replication group.
            for value in &mut self.state.values[previous_values..] {
                if parameter_origins.contains(&value.origin) {
                    value.storage_group = value.id;
                    value.owners = crate::tensor::OwnerMap::default();
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
                        self.config.capacity_baseline,
                    ),
                };
                let id = ensure_format(
                    id,
                    target,
                    OperandMaterialization::Complete,
                    operation.id,
                    "boundary",
                    self.costs,
                    &mut self.state,
                    &mut operations,
                );
                self.values.insert(output, id);
            }
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
                    site: None,
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
        // A body may select a compact native home for an invariant parameter.
        // Bind the outer allocation to that same format, just as for sequences.
        for (index, input) in inputs.iter_mut().enumerate().skip(repeat.carried_inputs) {
            let target = self.state.get(arguments[index]).tensor_type.format.clone();
            *input = ensure_format(
                *input,
                target,
                OperandMaterialization::Complete,
                operation.id,
                LocalSite::from("repeat.invariant").at(index as u32),
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
                .enumerate()
                .map(|(element, &id)| {
                    ensure_format(
                        id,
                        target.clone(),
                        OperandMaterialization::Complete,
                        operation.id,
                        LocalSite::from("repeat.sequence")
                            .at(index as u32)
                            .at(element as u32),
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
                LocalSite::from("repeat.yield").at(index as u32),
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
            site: None,
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
            &Recipe::default(),
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
                &Recipe::default(),
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
