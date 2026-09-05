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
        let mut bindings = Vec::new();
        for tile in 0..self.tile_count {
            let mut carried = Vec::with_capacity(repeat.carried_inputs);
            for index in 0..repeat.carried_inputs {
                let Some(argument) = self.find_local_shard(repeat.body.arguments[index], tile)?
                else {
                    continue;
                };
                let initial = self.corresponding_shard(operation.inputs[index], argument)?;
                let yielded = self.corresponding_shard(repeat.body.yields[index], argument)?;
                let result = self.corresponding_shard(operation.results[index], argument)?;
                self.alias_shard(argument, initial);
                if yielded != argument {
                    self.shards[yielded.index() as usize].definition =
                        ShardDefinition::WritableAlias(argument);
                }
                self.alias_shard(result, initial);
                carried.push(RepeatCarried {
                    initial,
                    argument,
                    yielded,
                    result,
                });
            }
            let invariants = (0..repeat.invariant_inputs)
                .filter_map(|index| {
                    let input_index = repeat.carried_inputs + index;
                    let argument =
                        match self.find_local_shard(repeat.body.arguments[input_index], tile) {
                            Ok(Some(argument)) => argument,
                            Ok(None) => return None,
                            Err(error) => return Some(Err(error)),
                        };
                    Some(
                        self.corresponding_shard(operation.inputs[input_index], argument)
                            .map(|input| RepeatInvariant { input, argument }),
                    )
                })
                .collect::<ExpansionResult<_>>()?;
            let iterated = repeat
                .iterated_inputs
                .iter()
                .enumerate()
                .filter_map(|(index, values)| {
                    let argument = match self
                        .find_local_shard(repeat.body.arguments[expected_inputs + index], tile)
                    {
                        Ok(Some(argument)) => argument,
                        Ok(None) => return None,
                        Err(error) => return Some(Err(error)),
                    };
                    let inputs = values
                        .iter()
                        .map(|value| self.corresponding_shard(*value, argument))
                        .collect::<ExpansionResult<Vec<_>>>();
                    let inputs = match inputs {
                        Ok(inputs) => inputs,
                        Err(error) => return Some(Err(error)),
                    };
                    let (alignment, access_tail) = iterated_requirements[index];
                    let strides = inputs
                        .iter()
                        .map(|shard| self.shard_stride(*shard, alignment, access_tail))
                        .collect::<ExpansionResult<Vec<_>>>();
                    let strides = match strides {
                        Ok(strides) => strides,
                        Err(error) => return Some(Err(error)),
                    };
                    let Some(&stride_bytes) = strides.first() else {
                        return Some(Err(ExpansionError::InvalidIteratedBlocks(index)));
                    };
                    if strides.iter().any(|stride| *stride != stride_bytes) {
                        return Some(Err(ExpansionError::InvalidIteratedBlocks(index)));
                    }
                    Some(Ok(RepeatIterated {
                        inputs,
                        argument,
                        stride_bytes,
                        alignment,
                    }))
                })
                .collect::<ExpansionResult<_>>()?;
            bindings.push(BlockRepeatBinding {
                tile,
                carried,
                invariants,
                iterated,
            });
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
    if let MidOperationKind::Primitive(crate::Primitive::Compute {
        reuse_input: Some(index),
        ..
    }) = &operation.kind
    {
        return value_can_alias(operation.inputs[*index], target, operations);
    }
    let Some(plan) = operation.operator_plan() else {
        return false;
    };
    let indices = match &plan.requirements.output_aliasing {
        OutputAliasing::Fresh => return false,
        OutputAliasing::MayAliasInputs(indices) => indices.as_slice(),
    };
    indices.iter().any(|index| {
        operation
            .inputs
            .get(usize::from(*index))
            .is_some_and(|input| value_can_alias(*input, target, operations))
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
    !operations[definition..]
        .iter()
        .any(|operation| operation.inputs.contains(&carried))
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
            let requirement = operation
                .operator_plan()
                .and_then(|plan| plan.requirements.inputs.get(index))
                .or_else(|| operation.conversion_plan().map(|plan| &plan.input));
            if let Some(requirement) = requirement {
                alignment = alignment.max(requirement.alignment);
                access_tail = access_tail.max(requirement.access_tail_bytes);
            }
        }
    }
    (alignment, access_tail)
}
