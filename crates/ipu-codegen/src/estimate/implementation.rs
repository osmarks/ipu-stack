//! Cost concrete implementations once; share their executable fragments.

use super::*;
use crate::mid::*;
use std::sync::{Arc, OnceLock};

pub(super) type ImplementationKey = (OperatorPlan, Vec<TensorType>, TensorType);
pub(super) type ImplementationCache =
    HashMap<ImplementationKey, Arc<OnceLock<Option<Arc<ImplementationEstimate>>>>>;

pub(crate) struct ImplementationEstimate {
    pub program: Arc<MidProgram>,
    pub memory: MemoryEstimate,
    pub exchange: ExchangeFootprint,
}

pub(crate) fn implementation_estimate(
    plan: &OperatorPlan,
    inputs: &[TensorType],
    output: &TensorType,
) -> Option<Arc<ImplementationEstimate>> {
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
                exchange: ExchangeFootprint::default(),
                deferred_inputs: vec![None; inputs.len()],
                implementation: None,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
            memory: MemoryEstimate::default(),
        }],
        outputs: vec![result],
        values,
        ..ImplementationCandidate::default()
    };
    let program = crate::mid::implementation::build_blocks(&candidate).ok()?;
    let low = crate::lower_to_tiles(&program, false);
    let (peak, temporary) = crate::place::program_memory(&low).ok()?;
    let live = inputs
        .iter()
        .chain(std::iter::once(output))
        .fold(MemoryUsage::default(), |usage, tensor| {
            usage.saturating_add(tensor_memory(tensor))
        });
    let exchange = super::program::program_footprint(&program).ok()?;
    Some(Arc::new(ImplementationEstimate {
        memory: MemoryEstimate {
            live,
            temporary: temporary.conservative_tensor_usage(),
            peak: peak.conservative_tensor_usage(),
            exchange_row_bytes: exchange.estimated_row_bytes(),
            maximum_standard_temporary_allocation: temporary.maximum_standard_allocation,
        },
        program,
        exchange,
    }))
}
