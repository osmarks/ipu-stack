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
    Ok(baseline::select(graph, &config, costs, &recipe)?.program)
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
        let no_worse = self.cycles <= other.cycles
            && self
                .memory
                .objectives()
                .into_iter()
                .zip(other.memory.objectives())
                .all(|(a, b)| a <= b);
        no_worse && self != other
    }
}

pub(super) fn deferred_claims(operations: &[MidOperation]) -> BTreeSet<MidValueId> {
    operations
        .iter()
        .flat_map(|operation| operation.deferred_inputs().iter().flatten())
        .map(|input| input.producer)
        .collect()
}

pub(super) fn restore_unclaimed_deferred_costs(operations: &mut [MidOperation]) -> bool {
    let mut changed = false;
    let claims = deferred_claims(operations);
    for operation in operations {
        let Some(offer) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
        else {
            continue;
        };
        if operation
            .results
            .first()
            .is_some_and(|result| !claims.contains(result))
        {
            changed = true;
            operation.estimated_cycles = offer.unfused_cycles;
            operation.estimated_exchange_cycles = offer.unfused_exchange_cycles;
            if let MidOperationKind::Operator { plan, .. } = &mut operation.kind {
                plan.deferred_output = None;
            }
        }
    }
    changed
}

pub(super) fn operation_graph_inputs(operation: &Operation, graph: &ComputeGraph) -> Vec<ValueId> {
    let mut inputs = operation.inputs.clone();
    if let OperationKind::Repeat(repeat) = &operation.kind {
        for sequence in &repeat.iterated_inputs {
            inputs.extend(&graph.sequences()[sequence.index() as usize].values);
        }
    }
    inputs
}

pub(super) fn apply_selected_plan(
    operation: &Operation,
    output_shape: TensorShape,
    mut plan: OperatorPlan,
    cast_orders: &[bool],
    single_use_inputs: &[bool],
    costs: &impl CostModel,
    values: &mut BTreeMap<ValueId, MidValueId>,
    state: &mut LoweringState,
    operations: &mut Vec<MidOperation>,
) {
    let input_ids = operation
        .inputs
        .iter()
        .map(|value| values[value])
        .collect::<Vec<_>>();
    let original_input_ids = input_ids.clone();
    let mut source_types = Vec::with_capacity(input_ids.len());
    let mut converted = Vec::with_capacity(input_ids.len());
    for ((value, requirement), &cast_before) in input_ids
        .into_iter()
        .zip(&plan.requirements.inputs)
        .zip(cast_orders)
    {
        let conversion_start = operations.len();
        let converted_value = ensure_format(
            value,
            requirement.format.clone(),
            requirement.materialization,
            cast_before,
            operation.id,
            costs,
            state,
            operations,
        );
        let streamed_source = operations[conversion_start..]
            .last_mut()
            .and_then(|conversion| {
                let streamed = conversion.conversion_plan().is_some_and(|plan| {
                    plan.output.materialization == OperandMaterialization::DispatchSlices
                });
                if streamed {
                    conversion.estimated_cycles = 0;
                    conversion.estimated_exchange_cycles = 0;
                    conversion.inputs.first().copied()
                } else {
                    None
                }
            });
        let source_value = streamed_source.unwrap_or(converted_value);
        source_types.push(state.get(source_value).tensor_type.clone());
        converted.push(converted_value);
    }
    let result = state.value(
        operation.results[0],
        TensorType {
            shape: output_shape,
            format: plan.requirements.output.format.clone(),
        },
    );
    let converted_types = converted
        .iter()
        .map(|value| state.get(*value).tensor_type.clone())
        .collect::<Vec<_>>();
    let implementation =
        costs.implementation(&plan, &converted_types, &state.get(result).tensor_type);
    let mut operator_cycles = costs
        .operator_cycle_override(&plan, &converted_types, &state.get(result).tensor_type)
        .unwrap_or_else(|| {
            implementation
                .as_ref()
                .map_or(u64::MAX, |p| p.estimated_cycles)
        });
    let mut operator_exchange_cycles = implementation
        .as_ref()
        .map_or(0, |p| p.estimated_exchange_cycles);
    // Preliminary transition prices remain useful for custom planning models.
    // Normal detailed ranking evaluates the emitted region, including movement.
    for ((source, input), requirement) in source_types
        .iter()
        .zip(&converted_types)
        .zip(&plan.requirements.inputs)
    {
        if requirement.materialization == OperandMaterialization::DispatchSlices
            && source.format.layout != input.format.layout
        {
            let cost = costs.rearrangement_cost(
                &input.shape,
                input.format.precision,
                layout_conversion_strategy(&source.format.layout, &input.format.layout),
                &source.format.layout,
                &input.format.layout,
            );
            operator_cycles = operator_cycles.saturating_add(cost.cycles);
            operator_exchange_cycles =
                operator_exchange_cycles.saturating_add(cost.exchange_cycles);
        }
    }
    let mut deferred_inputs = vec![None; converted.len()];
    for (input_index, ((&original, &converted), requirement)) in original_input_ids
        .iter()
        .zip(&converted)
        .zip(&plan.requirements.inputs)
        .enumerate()
    {
        let conversion_is_streamed = original == converted
            || operations.iter().any(|candidate| {
                candidate.inputs.as_slice() == [original]
                    && candidate.results.as_slice() == [converted]
                    && candidate.conversion_plan().is_some_and(|conversion| {
                        conversion.output.materialization == OperandMaterialization::DispatchSlices
                    })
            });
        if !conversion_is_streamed
            || !single_use_inputs.get(input_index).copied().unwrap_or(false)
            || requirement.materialization != OperandMaterialization::DispatchSlices
        {
            continue;
        }
        let Some(producer_index) = operations
            .iter()
            .position(|candidate| candidate.results.as_slice() == [original])
        else {
            continue;
        };
        let Some(offered) = operations[producer_index]
            .operator_plan()
            .and_then(|producer| producer.deferred_output)
        else {
            continue;
        };
        let Some(&source) = operations[producer_index].inputs.get(offered.source_input) else {
            continue;
        };
        deferred_inputs[input_index] = Some(DeferredInputPlan {
            producer: original,
            source,
            transform: offered.transform,
        });
    }
    tracing::trace!(
        source = operation.id.index(),
        cycles = operator_cycles,
        dispatch = ?plan.dispatch,
        input_layouts = ?converted_types
            .iter()
            .map(|input| &input.format.layout)
            .collect::<Vec<_>>(),
        output_layout = ?state.get(result).tensor_type.format.layout,
        "costed operator plan"
    );
    if let Some(offer) = &mut plan.deferred_output {
        offer.unfused_cycles = operator_cycles;
        offer.unfused_exchange_cycles = operator_exchange_cycles;
        operator_cycles = 0;
        operator_exchange_cycles = 0;
    }
    operations.push(MidOperation {
        source: Some(operation.id),
        inputs: converted,
        results: vec![result],
        kind: MidOperationKind::Operator {
            plan,
            deferred_inputs,
            implementation,
        },
        estimated_cycles: operator_cycles,
        estimated_exchange_cycles: operator_exchange_cycles,
    });
    values.insert(operation.results[0], result);
}

// Refresh all recipes: applying a plan may fuse or remove earlier operations,
// so adding only the newest operation would retain obsolete conversion costs.
// Cached operator fragments avoid resolving/cloning the whole prefix here.
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

// Eligibility only: the region beam compares complete conversion sequences.
fn early_cast_format(input: &TensorType, target: &TensorFormat) -> Option<TensorFormat> {
    if input.format.precision != Precision::F16
        || !matches!(target.precision, Precision::F8F143 { .. })
    {
        return None;
    }
    let quantized = TensorFormat {
        precision: target.precision,
        layout: input.fp8_producer_layout(target)?,
    };
    (quantized.layout.order == target.layout.order
        || quantized.supports_micro_panel_exchange(target))
    .then_some(quantized)
}

pub(super) fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    cast_before: bool,
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
    if packed_cast && !cast_before && target.layout.order == ElementOrder::Amp(AmpOrder::Left) {
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
    let early_cast = cast_before
        .then(|| early_cast_format(&state.get(value).tensor_type, &target))
        .flatten();
    let formats = if let Some(quantized) = early_cast {
        vec![quantized, target]
    } else {
        vec![
            TensorFormat {
                precision: from,
                layout: cast_layout.clone(),
            },
            TensorFormat {
                precision: target.precision,
                layout: if packed_cast && !cast_before {
                    target.layout.clone()
                } else {
                    cast_layout
                },
            },
            target,
        ]
    };
    for format in formats {
        let input = state.get(value).tensor_type.clone();
        if input.format == format {
            continue;
        }
        let output = TensorType {
            shape: input.shape.clone(),
            format,
        };
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
                input: OperandRequirement::new(input.format, 8),
                output: OperandRequirement::new(output.format, 8).with_materialization(if cast {
                    OperandMaterialization::Complete
                } else {
                    materialization
                }),
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
