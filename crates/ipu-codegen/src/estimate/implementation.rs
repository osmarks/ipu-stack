//! Cost concrete implementations once; share their executable fragments.

use super::*;
use crate::mid::*;
use std::sync::{Arc, Weak};

pub(super) type ImplementationKey = (OperatorPlan, Vec<TensorType>, TensorType);
pub(super) type ImplementationCache = HashMap<ImplementationKey, Weak<MidProgram>>;

pub(crate) fn implementation_estimate(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorType,
) -> Option<Arc<MidProgram>> {
    let values = inputs
        .iter()
        .chain(std::iter::once(output))
        .enumerate()
        .map(|(index, tensor)| {
            let id = MidValueId::from_index(index as u32);
            MidValue {
                id,
                tile_offset: 0,
                tensor_type: tensor.clone(),
                origin: crate::ValueId::from_index(index as u32),
                storage_group: id,
            }
        })
        .collect::<Vec<_>>();
    let result = values.last()?.id;
    let candidate = ImplementationCandidate {
        tile_count: values
            .iter()
            .map(|value| value.tensor_type.format.layout.tiling.tile_count)
            .max()?,
        inputs: values[..inputs.len()]
            .iter()
            .map(|value| MidInput {
                name: String::new(),
                kind: crate::GraphInputKind::Host,
                value: value.id,
            })
            .collect(),
        operations: vec![MidOperation {
            source: Some(crate::OperationId::from_index(0)),
            inputs: values[..inputs.len()]
                .iter()
                .map(|value| value.id)
                .collect(),
            results: vec![result],
            kind: MidOperationKind::Operator {
                plan: plan.clone(),
                deferred_inputs: vec![None; inputs.len()],
                implementation: None,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }],
        outputs: vec![result],
        values,
        ..ImplementationCandidate::default()
    };
    crate::mid::implementation::build_blocks(&candidate).ok()
}
