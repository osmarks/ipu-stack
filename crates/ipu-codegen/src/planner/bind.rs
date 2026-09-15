//! Apply whole-device selections and insert explicit format conversions.

use crate::compile::PipelineConfig;
use crate::estimate::{CostModel, MemoryUsage};
use crate::graph::{Operation, OperationId, OperationKind, ValueId};
use crate::kernel::TileKernelSpec;
use crate::low::default_copy_policy;
use crate::mid::{
    Compute, CoordinateMapping, LocalSite, MidOperation, MidOperationKind, MidValue, MidValueId,
    OperandIndexing, cast_order,
};

use crate::planner::cache::FragmentCache;
use crate::planner::error::{LoweringError, LoweringResult};
use crate::planner::operator::{OperandMaterialization, OperatorPlan};

use crate::tensor::{
    AmpOrder, ElementOrder, Layout, Precision, TensorFormat, TensorShape, TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Clone)]
pub(super) struct ValueBuilder {
    pub(super) values: Vec<MidValue>,
    pub(super) automatic_inputs: BTreeSet<MidValueId>,
    pub(super) parameter_values: BTreeSet<MidValueId>,
    pub(super) conversion_cycles: u64,
}

impl ValueBuilder {
    pub(super) fn value(&mut self, origin: ValueId, tensor_type: TensorType) -> MidValueId {
        let id = MidValueId::from_index(self.values.len() as u32);
        self.values.push(MidValue {
            id,
            owners: crate::tensor::OwnerMap::default(),
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
        &self.values[id.index() as usize]
    }

    pub(super) fn derived_value(
        &mut self,
        source: MidValueId,
        tensor_type: TensorType,
    ) -> MidValueId {
        let origin = self.get(source).origin;
        let storage_group = self.get(source).storage_group;
        let owners = self.get(source).owners.clone();
        let result = self.value_in_storage_group(origin, tensor_type, storage_group);
        self.values[result.index() as usize].owners = owners;
        if self.parameter_values.contains(&source) {
            self.parameter_values.insert(result);
        }
        result
    }

    pub(super) fn retarget_automatic_input(&mut self, id: MidValueId, layout: Layout) -> bool {
        if !self.automatic_inputs.remove(&id) {
            return false;
        }
        self.values[id.index() as usize].tensor_type.format.layout = layout;
        true
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
    fragments: &FragmentCache,
    tile_count: u16,
    values: &mut BTreeMap<ValueId, MidValueId>,
    state: &mut ValueBuilder,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<()> {
    let inputs = operation
        .inputs
        .iter()
        .map(|value| values[value])
        .collect::<Vec<_>>();
    let mut bound = Vec::with_capacity(inputs.len());
    for (index, (&input, requirement)) in inputs.iter().zip(&plan.requirements.inputs).enumerate() {
        bound.push(ensure_format(
            input,
            requirement.format.clone(),
            requirement.materialization,
            operation.id,
            LocalSite::from("input").at(index as u32),
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
    let fragment = fragments
        .get(plan, &input_types, &output)
        .ok_or(LoweringError::InvalidImplementation)?;
    let result = state.value(operation.results[0], output);
    if matches!(
        operation.kind,
        OperationKind::View(_) | OperationKind::Slice(_)
    ) && inputs.iter().any(|id| state.parameter_values.contains(id))
    {
        state.parameter_values.insert(result);
    }
    crate::mid::append_fragment(
        &fragment,
        &bound,
        &[result],
        &crate::tensor::OwnerMap::default(),
        Some(operation.id),
        tile_count,
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
        && matches!(&operation.kind, MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Cast { .. }, operands, output_aliases, ..
        }) if operands.len() == 1 && operands[0] == (OperandIndexing::Elementwise { result: 0 }) && output_aliases.is_empty()))
    .then(|| operation.results[0])
}

pub(super) fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    source: OperationId,
    site: impl Into<LocalSite>,
    costs: &impl CostModel,
    state: &mut ValueBuilder,
    operations: &mut Vec<MidOperation>,
) -> MidValueId {
    let site = site.into();
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
        if cast_order::producer_layout(&row_type, &target).is_some() {
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
                        default_copy_policy(&input.format.layout, layout),
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
        let policy = default_copy_policy(&input.format.layout, &output.format.layout);
        let cycles = if cast {
            costs.cast_format_cycles(&input, &output.format)
        } else {
            costs
                .rearrangement_cost(
                    &output.shape,
                    output.format.precision,
                    policy,
                    &input.format.layout,
                    &output.format.layout,
                )
                .cycles
        };
        state.conversion_cycles = state.conversion_cycles.saturating_add(cycles);
        let result = state.derived_value(value, output.clone());
        operations.push(MidOperation {
            site: Some(site.child(["input.copy", "cast", "output.copy"][index])),
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: if cast {
                MidOperationKind::Compute(Compute::cast(
                    input.format.precision,
                    output.format.precision,
                ))
            } else {
                MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                    policy,
                    packing: crate::PackingPolicy::Automatic,
                }
            },
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
