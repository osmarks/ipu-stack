//! Structured repeat bodies and carried/iterated block bindings.

use super::*;

impl TileGraphBuilder {
    pub(super) fn build_repeat(
        &mut self,
        operation: &MidOperation,
        repeat: &MidRepeat,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let expected_inputs = repeat.carried_inputs + repeat.invariant_inputs;
        let expected_arguments = expected_inputs + repeat.iterated_inputs.len();
        if operation.inputs.len() != expected_inputs
            || operation.results.len() != repeat.carried_inputs
            || repeat.body.arguments.len() != expected_arguments
            || repeat.body.yields.len() != repeat.carried_inputs
            || repeat
                .iterated_inputs
                .iter()
                .any(|values| values.len() != repeat.count as usize)
        {
            return Err(ExpansionError::InvalidRepeat);
        }
        for index in 0..repeat.carried_inputs {
            if !repeat_yield_can_alias(
                repeat.body.yields[index],
                repeat.body.arguments[index],
                &repeat.body.operations,
            ) {
                return Err(ExpansionError::RepeatRequiresInPlace(index));
            }
        }
        let iterated_requirements = repeat
            .iterated_inputs
            .iter()
            .enumerate()
            .map(|(index, _)| {
                body_storage_requirement(
                    repeat.body.arguments[expected_inputs + index],
                    &repeat.body.operations,
                )
            })
            .collect::<Vec<_>>();
        let body = self.build_region(&repeat.body.operations, false)?;
        let mut bindings = (0..self.tile_count)
            .map(|tile| BlockRepeatBinding {
                tile,
                carried: Vec::new(),
                invariants: Vec::new(),
                iterated: Vec::new(),
            })
            .collect::<Vec<_>>();
        for index in 0..repeat.carried_inputs {
            for argument in self.value_shards(repeat.body.arguments[index])?.to_vec() {
                let initial = self.corresponding_shard(operation.inputs[index], argument)?;
                let yielded = self.corresponding_shard(repeat.body.yields[index], argument)?;
                let result = self.corresponding_shard(operation.results[index], argument)?;
                self.alias_shard(argument, initial);
                if yielded != argument {
                    self.shards[yielded.index() as usize].definition =
                        ShardDefinition::WritableAlias(argument);
                }
                self.alias_shard(result, initial);
                let tile = self.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)].carried.push(RepeatCarried {
                    initial,
                    argument,
                    yielded,
                    result,
                });
            }
        }
        for index in repeat.carried_inputs..expected_inputs {
            for &argument in self.value_shards(repeat.body.arguments[index])? {
                let tile = self.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)]
                    .invariants
                    .push(RepeatInvariant {
                        input: self.corresponding_shard(operation.inputs[index], argument)?,
                        argument,
                    });
            }
        }
        for (index, values) in repeat.iterated_inputs.iter().enumerate() {
            for &argument in self.value_shards(repeat.body.arguments[expected_inputs + index])? {
                let inputs = values
                    .iter()
                    .map(|value| self.corresponding_shard(*value, argument))
                    .collect::<ExpansionResult<Vec<_>>>()?;
                let (alignment, access_tail) = iterated_requirements[index];
                let mut strides = inputs
                    .iter()
                    .map(|shard| self.shard_stride(*shard, alignment, access_tail));
                let stride_bytes = strides
                    .next()
                    .ok_or(ExpansionError::InvalidIteratedBlocks(index))??;
                for stride in strides {
                    if stride? != stride_bytes {
                        return Err(ExpansionError::InvalidIteratedBlocks(index));
                    }
                }
                let tile = self.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)].iterated.push(RepeatIterated {
                    inputs,
                    argument,
                    stride_bytes,
                    alignment,
                });
            }
        }
        tiles
            .operations
            .push(BlockOperation::Repeat(Box::new(BlockRepeat {
                provenance: WorkProvenance {
                    operation: operation.source,
                    value: operation.results.first().copied(),
                    reason: WorkReason::Repeat,
                },
                count: repeat.count,
                bindings,
                body,
            })));
        Ok(())
    }
}

fn value_can_alias(value: MidValueId, target: MidValueId, operations: &[MidOperation]) -> bool {
    if value == target {
        return true;
    }
    let Some(operation) = operations
        .iter()
        .find(|operation| operation.results.contains(&value))
    else {
        return false;
    };
    if let MidOperationKind::Primitive(crate::Primitive::Compute { output_aliases, .. }) =
        &operation.kind
    {
        return output_aliases.iter().any(|&(output, input)| {
            operation.results[output] == value
                && value_can_alias(operation.inputs[input], target, operations)
        });
    }
    false
}

fn repeat_yield_can_alias(
    value: MidValueId,
    carried: MidValueId,
    operations: &[MidOperation],
) -> bool {
    if value_can_alias(value, carried, operations) {
        return true;
    }
    let Some(definition) = operations
        .iter()
        .position(|operation| operation.results.contains(&value))
    else {
        return false;
    };
    // A repeat reuses the carried allocation on its next iteration. A fresh
    // yield may overwrite it when every read of the previous iteration's
    // value has completed before the yielding operation begins.
    let mut aliases = BTreeSet::from([carried]);
    for (index, operation) in operations.iter().enumerate() {
        if index >= definition && operation.read_values().any(|input| aliases.contains(input)) {
            return false;
        }
        match &operation.kind {
            MidOperationKind::Primitive(crate::Primitive::Copy {
                reuse_local: true, ..
            }) if operation.inputs.iter().any(|input| aliases.contains(input)) => {
                // Internal copies may become local views during expansion.
                aliases.extend(operation.results.iter().copied());
            }
            MidOperationKind::Primitive(crate::Primitive::Compute { output_aliases, .. }) => {
                for &(output, input) in output_aliases {
                    if aliases.contains(&operation.inputs[input]) {
                        aliases.insert(operation.results[output]);
                    }
                }
            }
            _ => {}
        }
    }
    true
}

fn body_storage_requirement(value: MidValueId, operations: &[MidOperation]) -> (u32, u32) {
    let mut alignment = 8;
    let mut access_tail = 0;
    for operation in operations {
        for (index, input) in operation.inputs.iter().enumerate() {
            if *input != value {
                continue;
            }
            if let MidOperationKind::Primitive(crate::Primitive::Compute {
                kernel: TileKernelSpec::Gemm { multiply, .. },
                ..
            }) = &operation.kind
            {
                alignment = alignment.max(32);
                if index == 0 {
                    access_tail = access_tail.max(8 * multiply.bytes() as u32);
                }
            }
        }
    }
    (alignment, access_tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoordinateMapping, OperandWindow, Primitive};

    #[test]
    fn carried_storage_remains_live_through_reused_copy_inputs() {
        let id = MidValueId::from_index;
        let op = |input, output, kind| MidOperation {
            source: None,
            inputs: vec![id(input)],
            results: vec![id(output)],
            kind: MidOperationKind::Primitive(kind),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let gelu = || Primitive::Compute {
            kernel: TileKernelSpec::Gelu,
            operands: vec![OperandWindow::default()],
            product: None,
            output_aliases: vec![],
        };
        for reuse_local in [false, true] {
            let operations = vec![
                op(
                    0,
                    2,
                    Primitive::Copy {
                        mapping: CoordinateMapping::default(),
                        reuse_local,
                    },
                ),
                op(1, 3, gelu()),
                op(2, 4, gelu()),
            ];
            assert_eq!(
                repeat_yield_can_alias(id(3), id(0), &operations),
                !reuse_local,
                "the later reader needs the previous carried value unless its copy was materialized"
            );
        }
    }
}
