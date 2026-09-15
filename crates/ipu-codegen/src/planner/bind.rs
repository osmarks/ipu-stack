//! Apply whole-device selections and insert explicit format conversions.

use crate::PipelineConfig;
use crate::estimate::CostModel;
use crate::graph::{Operation, OperationId, OperationKind, ValueId};
use crate::mid::MidOperationKind;

use crate::low::default_copy_policy;
use crate::mid::{
    CoordinateMapping, MidOperation, MidValue, MidValueId, OperandIndexing, cast_order,
};

use crate::planner::cache::FragmentCache;
use crate::planner::error::{LoweringError, LoweringResult};
use crate::planner::operator::{OperandMaterialization, OperatorPlan};

use crate::tensor::{
    AmpOrder, AxisTiling, ElementOrder, Layout, Padding, Precision, TensorAxis, TensorFormat,
    TensorShape, TensorTiling, TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default, Clone)]
pub(super) struct ValueBuilder {
    pub(super) values: Vec<MidValue>,
    pub(super) automatic_inputs: BTreeSet<MidValueId>,
    pub(super) parameter_values: BTreeSet<MidValueId>,
    pub(super) copies: BTreeMap<MidValueId, u32>,
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
        let result = self.value(origin, tensor_type);
        self.values[result.index() as usize].storage_group = storage_group;
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
    inputs: &[MidValueId],
    output_shape: TensorShape,
    plan: &OperatorPlan,
    open_boundary: bool,
    config: &PipelineConfig,
    costs: &impl CostModel,
    fragments: &FragmentCache,
    state: &mut ValueBuilder,
    operations: &mut Vec<MidOperation>,
) -> LoweringResult<Vec<MidValueId>> {
    // Persistent homes do not inherit the compute operand's replication.
    // Costing and insertion use this same choice, including Repeat multiplicity.
    for (&id, requirement) in inputs.iter().zip(&plan.inputs) {
        if state.automatic_inputs.contains(&id) && state.parameter_values.contains(&id) {
            let mut tensor = state.get(id).tensor_type.clone();
            tensor.format.layout.order = requirement.format.layout.order;
            let layout = super::parameter_homes::compact_parameter_layout(
                &tensor,
                state.copies.get(&id).copied().unwrap_or(1),
                config,
            )
            .ok_or(LoweringError::NoCandidate(operation.id))?;
            state.retarget_automatic_input(id, layout);
        }
    }
    let parameter_origins = inputs
        .iter()
        .filter(|id| state.parameter_values.contains(id))
        .map(|id| state.get(*id).origin)
        .collect::<BTreeSet<_>>();
    let previous_values = state.values.len();
    let mut bound = Vec::with_capacity(inputs.len());
    for (&input, requirement) in inputs.iter().zip(&plan.inputs) {
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
        format: plan.output.clone(),
    };
    let input_types = bound
        .iter()
        .map(|&id| state.get(id).tensor_type.clone())
        .collect::<Vec<_>>();
    let fragment = fragments
        .get(plan, &input_types, &output)
        .ok_or(LoweringError::InvalidImplementation)?;
    let mut results = crate::mid::append_fragment(
        &fragment,
        &bound,
        &crate::tensor::OwnerMap::default(),
        Some(operation.id),
        operation.results[0],
        config.tile_count,
        &mut state.values,
        operations,
    )
    .ok_or(LoweringError::InvalidImplementation)?;
    if matches!(
        operation.kind,
        OperationKind::View(_) | OperationKind::Slice(_)
    ) && inputs.iter().any(|id| state.parameter_values.contains(id))
    {
        state.parameter_values.extend(&results);
    }
    // Parameter conversions are temporaries, not persistent sequence members.
    for value in &mut state.values[previous_values..] {
        if parameter_origins.contains(&value.origin) {
            value.storage_group = value.id;
            value.owners = crate::tensor::OwnerMap::default();
        }
    }
    if !open_boundary {
        for result in &mut results {
            let tensor = &state.get(*result).tensor_type;
            *result = ensure_format(
                *result,
                TensorFormat {
                    precision: tensor.format.precision,
                    layout: canonical(
                        &tensor.shape,
                        tensor.format.precision,
                        config.tile_count,
                        config.capacity_baseline,
                    ),
                },
                OperandMaterialization::Complete,
                operation.id,
                costs,
                state,
                operations,
            );
        }
    }
    Ok(results)
}

pub(super) fn canonical(
    shape: &TensorShape,
    precision: Precision,
    tiles: u16,
    capacity: bool,
) -> Layout {
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

pub(super) fn flat(shape: &TensorShape, precision: Precision, tiles: u16) -> Layout {
    let mut grain = (8 / precision.bytes()) as u32;
    while !shape.elements().is_multiple_of(u64::from(grain)) {
        grain /= 2;
    }
    let owners = (shape.elements() / u64::from(grain)).min(u64::from(tiles)) as u16;
    Layout::logical_linear(owners, grain)
}

fn reusable_cast(operation: &MidOperation, input: MidValueId) -> Option<MidValueId> {
    (operation.inputs.as_slice() == [input]
        && matches!(operation, MidOperation {
kind: MidOperationKind::Cast { .. },
operands,
output_aliases,
..
} if operands.len() == 1 && operands[0] == (OperandIndexing::Elementwise { result: 0 }) && output_aliases.is_empty()))
    .then(|| operation.results[0])
}

pub(super) fn ensure_format(
    mut value: MidValueId,
    target: TensorFormat,
    materialization: OperandMaterialization,
    source: OperationId,

    costs: &impl CostModel,
    state: &mut ValueBuilder,
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
            source: Some(source),
            inputs: vec![value],
            results: vec![result],
            kind: if cast {
                MidOperationKind::Cast {
                    from: input.format.precision,
                    to: output.format.precision,
                }
            } else {
                MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                    policy,
                    packing: crate::PackingPolicy::Automatic,
                }
            },
            operands: if cast {
                vec![OperandIndexing::Elementwise { result: 0 }]
            } else {
                vec![]
            },
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
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
