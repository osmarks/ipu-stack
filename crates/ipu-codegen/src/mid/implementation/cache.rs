//! Compilation-scoped reuse of executable family fragments. Cost models consume
//! these programs; they neither construct them nor own this cache. Keys include
//! the actual boundary types, including layout and precision.

use super::implement;
use crate::mid::{MidProgram, OperatorPlan};
use crate::tensor::TensorType;
use foldhash::fast::FixedState;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type Key = (OperatorPlan, Vec<TensorType>, TensorType);
type Entry = Arc<OnceLock<Option<Arc<MidProgram>>>>;

#[derive(Default)]
pub(crate) struct FragmentCache {
    entries: Mutex<HashMap<Key, Entry, FixedState>>,
}

impl FragmentCache {
    pub(crate) fn get(
        &self,
        plan: &OperatorPlan,
        inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<Arc<MidProgram>> {
        let key = (plan.clone(), inputs.to_vec(), output.clone());
        let entry = self.entries.lock().unwrap().entry(key).or_default().clone();
        entry
            .get_or_init(|| implement(plan, inputs, output))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConcreteOperatorCandidate, Layout, MidOperator, OperandRequirement, Precision, TensorFormat,
    };
    use rayon::prelude::*;

    #[test]
    fn concurrent_fragments_keep_their_actual_boundary_contracts() {
        let cache = FragmentCache::default();
        let output = TensorType::new([16, 64], Precision::F16, Layout::row_sharded(4));
        let plan = ConcreteOperatorCandidate::new(
            MidOperator::Gelu,
            [OperandRequirement::new(output.format.clone())],
            OperandRequirement::new(output.format.clone()),
        )
        .plan;
        let variants = (1..=4)
            .map(|tiles| TensorType {
                shape: output.shape.clone(),
                format: TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::row_sharded(tiles),
                },
            })
            .collect::<Vec<_>>();
        (0..32).into_par_iter().for_each(|index| {
            let inputs = std::slice::from_ref(&variants[index % variants.len()]);
            let expected = implement(&plan, inputs, &output).unwrap();
            let retained = cache.get(&plan, inputs, &output).unwrap();
            assert_eq!(retained, expected);
            retained.validate().unwrap();
            let input = retained.inputs[0].value;
            assert_eq!(
                retained.values[input.index() as usize].tensor_type,
                inputs[0]
            );
        });
    }
}
