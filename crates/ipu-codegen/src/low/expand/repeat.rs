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
        let body = self.build_region(&repeat.body.operations, false)?;
        let mut bindings = (0..self.program.tile_count)
            .map(|tile| BlockRepeatBinding {
                tile,
                carried: Vec::new(),
                invariants: Vec::new(),
                iterated: Vec::new(),
            })
            .collect::<Vec<_>>();
        for index in 0..repeat.carried_inputs {
            for argument in self
                .allocation_shards(repeat.body.arguments[index])?
                .to_vec()
            {
                let initial = self.corresponding_shard(operation.inputs[index], argument)?;
                let yielded = self.corresponding_shard(repeat.body.yields[index], argument)?;
                let result = self.corresponding_shard(operation.results[index], argument)?;
                self.program.shards[argument.index() as usize].definition =
                    ShardDefinition::Alias(initial);
                if yielded != argument {
                    self.program.shards[yielded.index() as usize].definition =
                        ShardDefinition::WritableAlias(argument);
                }
                self.program.shards[result.index() as usize].definition =
                    ShardDefinition::Alias(initial);
                let tile = self.program.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)].carried.push(RepeatCarried {
                    initial,
                    argument,
                    yielded,
                    result,
                });
            }
        }
        for index in repeat.carried_inputs..expected_inputs {
            for argument in self.allocation_shards(repeat.body.arguments[index])? {
                let tile = self.program.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)]
                    .invariants
                    .push(RepeatInvariant {
                        input: self.corresponding_shard(operation.inputs[index], argument)?,
                        argument,
                    });
            }
        }
        for (index, values) in repeat.iterated_inputs.iter().enumerate() {
            for argument in
                self.allocation_shards(repeat.body.arguments[expected_inputs + index])?
            {
                let inputs = values
                    .iter()
                    .map(|value| self.corresponding_shard(*value, argument))
                    .collect::<ExpansionResult<Vec<_>>>()?;
                let tile = self.program.shards[argument.index() as usize].tile;
                bindings[usize::from(tile)]
                    .iterated
                    .push(RepeatIterated { inputs, argument });
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
    fn corresponding_shard(
        &self,
        value: MidValueId,
        target: BlockValueId,
    ) -> ExpansionResult<BlockValueId> {
        let target = &self.program.shards[target.index() as usize];
        self.allocation_shards(value)?
            .into_iter()
            .filter(|shard| self.program.shards[shard.index() as usize].extents == target.extents)
            .min_by_key(|shard| {
                u8::from(self.program.shards[shard.index() as usize].tile != target.tile)
            })
            .ok_or(ExpansionError::UnknownValue(value))
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
    operation
        .output_aliases
        .as_slice()
        .iter()
        .any(|&(output, input)| {
            operation.results[output] == value
                && value_can_alias(operation.inputs[input], target, operations)
        })
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
        for &(output, input) in &operation.output_aliases {
            if aliases.contains(&operation.inputs[input]) {
                aliases.insert(operation.results[output]);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CoordinateMapping, OperandIndexing};

    #[test]
    fn carried_storage_remains_live_through_reused_copy_inputs() {
        let id = MidValueId::from_index;
        let op = |input, output, kind| MidOperation {
            source: None,
            inputs: vec![id(input)],
            results: vec![id(output)],
            operands: if matches!(kind, MidOperationKind::Gelu) {
                vec![OperandIndexing::Elementwise { result: 0 }]
            } else {
                vec![]
            },
            kind,
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
        };
        let gelu = || MidOperationKind::Gelu;
        for view in [false, true] {
            let mut operations = vec![
                op(
                    0,
                    2,
                    MidOperationKind::Copy {
                        policy: crate::CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Staged,
                        mapping: CoordinateMapping::default(),
                    },
                ),
                op(1, 3, gelu()),
                op(if view { 0 } else { 2 }, 4, gelu()),
            ];
            if view {
                operations.remove(0);
            }
            assert_eq!(
                repeat_yield_can_alias(id(3), id(0), &operations),
                !view,
                "the later reader needs the previous carried value unless its copy was materialized"
            );
        }
    }
}
