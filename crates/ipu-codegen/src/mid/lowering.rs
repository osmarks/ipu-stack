//! Apply whole-device selections and insert explicit format conversions.

use super::*;

#[cfg(test)]
pub fn lower(
    graph: &ComputeGraph,
    config: &PipelineConfig,
    costs: &impl CostModel,
) -> LoweringResult<MidProgram> {
    // Operator/kernel tests exercise a neighborhood with exposed boundaries.
    fn outputs(ops: &[Operation], ids: &mut BTreeSet<ValueId>) {
        for op in ops {
            ids.extend(&op.results);
            if let OperationKind::Repeat(repeat) = &op.kind {
                outputs(&repeat.body.operations, ids);
            }
        }
    }
    // These fixtures test operator lowering with explicit distributed host
    // bindings, independently of the package baseline's coarser boundary policy.
    let mut config = config.clone();
    for input in graph
        .inputs()
        .iter()
        .filter(|input| input.kind == GraphInputKind::Host)
    {
        if let Some(&precision) = config.automatic_inputs.get(&input.value) {
            let rows = input
                .shape
                .0
                .get(input.shape.0.len().saturating_sub(2))
                .copied()
                .unwrap_or(1);
            config.inputs.insert(
                input.value,
                TensorFormat {
                    precision,
                    layout: Layout::row_sharded(u32::from(config.tile_count).min(rows) as u16),
                },
            );
        }
    }
    let mut recipe = baseline::Recipe::default();
    outputs(graph.operations(), &mut recipe.open_boundaries);
    recipe
        .early_casts
        .extend(graph.operations().iter().map(|op| op.id));
    let mut program = baseline::select(graph, &config, costs, &recipe)?.program;
    program.compose_copies();
    program
        .refresh_estimates()
        .ok_or(LoweringError::InvalidImplementation)?;
    Ok(program)
}

#[derive(Default, Clone)]
pub(super) struct LoweringState {
    pub(super) values: Vec<MidValue>,
    pub(super) automatic_inputs: BTreeSet<MidValueId>,
    pub(super) parameter_values: BTreeSet<MidValueId>,
}

impl LoweringState {
    pub(super) fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            tile_offset: 0,
            tensor_type,
            origin,
            storage_group: id,
        });
        id
    }

    pub(super) fn value_in_storage_group(
        &mut self,
        origin: ValueId,
        tensor_type: TensorType,
        storage_group: MidValueId,
    ) -> MidValueId {
        let result = self.value(origin, tensor_type);
        self.values[result.index() as usize].storage_group = storage_group;
        result
    }

    pub(super) fn get(&self, id: MidValueId) -> &MidValue {
        &self.values[id.0 as usize]
    }

    pub(super) fn derived_value(
        &mut self,
        source: MidValueId,
        tensor_type: TensorType,
    ) -> MidValueId {
        let origin = self.get(source).origin;
        let storage_group = self.get(source).storage_group;
        let offset = self.get(source).tile_offset;
        let result = self.value_in_storage_group(origin, tensor_type, storage_group);
        self.values[result.index() as usize].tile_offset = offset;
        if self.parameter_values.contains(&source) {
            self.parameter_values.insert(result);
        }
        result
    }

    pub(super) fn retarget_automatic_input(&mut self, id: MidValueId, layout: Layout) -> bool {
        if !self.automatic_inputs.remove(&id) {
            return false;
        }
        self.values[id.0 as usize].tensor_type.format.layout = layout;
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlanMetrics {
    pub(super) cycles: u64,
    pub(super) memory: MemoryPeaks,
}

impl PlanMetrics {
    pub(super) fn dominates(self, other: Self) -> bool {
        let memory = self.memory.objectives();
        let other_memory = other.memory.objectives();
        let no_worse = self.cycles <= other.cycles
            && memory.into_iter().zip(other_memory).all(|(a, b)| a <= b);
        no_worse && (self.cycles < other.cycles || memory != other_memory)
    }
}

/// Construct the selected family directly from the values available in this
/// region. A panel requirement leaves layout movement to that family's copies;
/// casts and complete operands are materialized before binding the fragment.
pub(super) fn emit_selected(
    operation: &Operation,
    output_shape: TensorShape,
    plan: &OperatorPlan,
    costs: &impl CostModel,
    values: &mut BTreeMap<ValueId, MidValueId>,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| values[value])
        .collect::<Vec<_>>();
    let mut bound = Vec::with_capacity(inputs.len());
    for (&input, requirement) in inputs.iter().zip(&plan.requirements.inputs) {
        bound.push(ensure_format(
            input,
            requirement.format.clone(),
            requirement.materialization,
            operation.id,
            costs,
            state,
            operations,
        ));
    }
    let output = TensorType {
        shape: output_shape,
        format: plan.requirements.output.format.clone(),
    };
    let input_types = bound
        .iter()
        .map(|&id| state.get(id).tensor_type.clone())
        .collect::<Vec<_>>();
    let fragment = costs
        .implementation(plan, &input_types, &output)
        .ok_or(LoweringError::InvalidImplementation)?;
    let result = state.value(operation.results[0], output);
    if matches!(
        operation.kind,
        OperationKind::View(_) | OperationKind::Slice(_)
    ) && inputs.iter().any(|id| state.parameter_values.contains(id))
    {
        state.parameter_values.insert(result);
    }
    implementation::append_fragment(
        &fragment,
        &bound,
        &[result],
        Some(operation.id),
        &mut state.values,
        operations,
    )
    .ok_or(LoweringError::InvalidImplementation)?;
    values.insert(operation.results[0], result);
    Ok(())
}

// Cheap necessary memory check for individual operands. The executable
// fragment's liveness analysis accounts for simultaneous operands and scratch.
pub(super) fn plan_fits_operator_memory(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorShape,
    config: &PipelineConfig,
) -> bool {
    let (planned_inputs, planned_output) = plan.tensor_types(inputs, output);
    let peak = planned_inputs
        .iter()
        .chain(std::iter::once(&planned_output))
        .map(crate::estimate::tensor_memory)
        .fold(MemoryUsage::default(), |peak, tensor| MemoryUsage {
            standard: peak.standard.max(tensor.standard),
            interleaved: peak.interleaved.max(tensor.interleaved),
        });
    peak.interleaved <= u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES)
        && peak
            .total()
            .saturating_add(config.standard_memory_reservation_bytes)
            <= config
                .tile_memory_budget_bytes
                .min(u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES))
}

fn reusable_cast(operation: &MidOperation, input: MidValueId) -> Option<MidValueId> {
    (operation.inputs.as_slice() == [input]
        && operation.conversion_plan().is_some_and(|plan| {
            plan.strategy == ConversionStrategy::LocalKernel
                && plan.input.format.precision != plan.output.format.precision
                && plan.output.materialization == OperandMaterialization::Complete
        }))
    .then(|| operation.results[0])
}

pub(super) fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    source: OperationId,
    costs: &impl CostModel,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    let from = state.get(value).tensor_type.format.precision;
    let fp8_cast = from != target.precision
        && (matches!(from, Precision::F8F143 { .. })
            || matches!(target.precision, Precision::F8F143 { .. }));
    // The F16-to-FP8 kernel regroups pairs of 16-element panels directly
    // into 32-element panels. Redistribute in the packed F16 order first;
    // no row-major intermediate or subsequent byte-sized transpose is needed.
    let packed_cast =
        from == Precision::F16 && matches!(target.precision, Precision::F8F143 { .. });
    let mut initial_layout = target.layout.clone();
    if fp8_cast && !packed_cast {
        initial_layout.order = ElementOrder::RowMajor;
    }
    if state.retarget_automatic_input(value, initial_layout.clone())
        && state.get(value).tensor_type.format.precision == target.precision
    {
        return value;
    }
    // For late FP8 conversion, compare packing F16 before the cast with
    // receiving row-major F16 and constructing the FP8 panels directly.
    // Ownership stays fixed; this chooses a local conversion implementation.
    if packed_cast && target.layout.order == ElementOrder::Amp(AmpOrder::Left) {
        let input = &state.get(value).tensor_type;
        let mut row_layout = initial_layout.clone();
        row_layout.order = ElementOrder::RowMajor;
        let row_type = TensorType {
            shape: input.shape.clone(),
            format: TensorFormat {
                precision: from,
                layout: row_layout.clone(),
            },
        };
        if row_type.fp8_producer_layout(&target).is_some() {
            let price = |layout: &Layout| {
                let staging = TensorType {
                    shape: input.shape.clone(),
                    format: TensorFormat {
                        precision: from,
                        layout: layout.clone(),
                    },
                };
                costs
                    .rearrangement_cost(
                        &input.shape,
                        from,
                        layout_conversion_strategy(&input.format.layout, layout),
                        &input.format.layout,
                        layout,
                    )
                    .cycles
                    .saturating_add(costs.cast_format_cycles(&staging, &target))
            };
            if price(&row_layout) < price(&initial_layout) {
                initial_layout = row_layout;
            }
        }
    }
    let cast_layout = if fp8_cast {
        initial_layout
    } else {
        state.get(value).tensor_type.format.layout.clone()
    };
    let formats = [
        TensorFormat {
            precision: from,
            layout: cast_layout.clone(),
        },
        TensorFormat {
            precision: target.precision,
            layout: if packed_cast {
                target.layout.clone()
            } else {
                cast_layout
            },
        },
        target,
    ];
    for (index, format) in formats.iter().enumerate() {
        let input = state.get(value).tensor_type.clone();
        if &input.format == format {
            continue;
        }
        let output = TensorType {
            shape: input.shape.clone(),
            format: format.clone(),
        };
        // A panel consumer emits its own copies from this source. A layout
        // needed by a following cast is complete, because that cast reads it.
        if materialization == OperandMaterialization::DispatchSlices
            && formats[index..]
                .iter()
                .all(|next| next.precision == input.format.precision)
        {
            continue;
        }
        // Share quantization on the producer's owners, not the much larger
        // replicated consumer operands whose lifetimes should remain local.
        if let Some(existing) = operations
            .iter()
            .rev()
            .filter_map(|operation| reusable_cast(operation, value))
            .find(|&id| state.get(id).tensor_type.format == output.format)
        {
            value = existing;
            continue;
        }
        let cast = input.format.precision != output.format.precision;
        let strategy = if cast {
            ConversionStrategy::LocalKernel
        } else {
            layout_conversion_strategy(&input.format.layout, &output.format.layout)
        };
        let cost = if cast {
            crate::estimate::RearrangementCost {
                cycles: costs.cast_format_cycles(&input, &output.format),
                ..Default::default()
            }
        } else {
            costs.rearrangement_cost(
                &output.shape,
                output.format.precision,
                strategy,
                &input.format.layout,
                &output.format.layout,
            )
        };
        let result = state.derived_value(value, output.clone());
        operations.push(MidOperation {
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: MidOperationKind::Convert(ConversionPlan {
                input: OperandRequirement::new(input.format),
                output: OperandRequirement::new(output.format),
                strategy,
            }),
            estimated_cycles: cost.cycles,
            estimated_exchange_cycles: cost.exchange_cycles,
        });
        value = result;
    }
    value
}

pub(super) fn lookup(
    values: &BTreeMap<ValueId, MidValueId>,
    value: ValueId,
) -> LoweringResult<MidValueId> {
    values
        .get(&value)
        .copied()
        .ok_or(LoweringError::UnknownValue(value))
}
