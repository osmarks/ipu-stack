//! Bind an already executable fragment into an enclosing mid region.
use super::{CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId};
use crate::graph::OperationId;
use crate::low::{CopyPolicy, PackingPolicy};
use std::collections::BTreeMap;

/// Substitute complete boundary bindings, preserving the fragment's operations
/// and ownership groups. Unbound groups rotate with the first result (or input);
/// groups shared with a boundary retain that boundary's mapping instead.
/// New temporaries belong to the caller's semantic result for diagnostics.
/// A returned input is connected to a distinct caller result by an explicit
/// identity copy, which ordinary copy lowering can reuse locally.
/// Failure leaves both caller vectors unchanged.
pub(crate) fn append_fragment(
    fragment: &MidProgram,
    inputs: &[MidValueId],
    outputs: &[MidValueId],
    source: Option<OperationId>,
    tile_count: u16,
    values: &mut Vec<MidValue>,
    operations: &mut Vec<MidOperation>,
) -> Option<()> {
    if tile_count == 0
        || fragment.tile_count > tile_count
        || fragment.inputs.len() != inputs.len()
        || fragment.outputs.len() != outputs.len()
    {
        return None;
    }
    fragment.validate().ok()?;
    let mut ids = vec![None; fragment.values.len()];
    let mut groups = vec![None; fragment.values.len()];
    let mut returned = Vec::new();
    let mut output_bindings = BTreeMap::new();
    for (&from, &to) in fragment.outputs.iter().zip(outputs) {
        if output_bindings
            .insert(to, from)
            .is_some_and(|old| old != from)
        {
            return None;
        }
    }
    let bindings = fragment
        .inputs
        .iter()
        .zip(inputs)
        .map(|(input, &to)| (true, input.value, to))
        .chain(
            fragment
                .outputs
                .iter()
                .zip(outputs)
                .map(|(&from, &to)| (false, from, to)),
        );
    for (input, from, to) in bindings {
        let template = &fragment.values[from.index() as usize];
        let actual = values.get(to.index() as usize)?;
        if actual.id != to
            || actual.tensor_type != template.tensor_type
            || actual.tile_offset >= tile_count
            || actual.storage_group.index() as usize >= values.len()
        {
            return None;
        }
        if let Some(existing) = ids[from.index() as usize] {
            if existing != to {
                if input || inputs.contains(&to) {
                    return None;
                }
                if !returned.contains(&(existing, to)) {
                    returned.push((existing, to));
                }
            }
            continue;
        }
        if !input && inputs.contains(&to) {
            return None; // A computed result cannot redefine an enclosing input.
        }
        let group = &mut groups[template.storage_group.index() as usize];
        let binding = (
            actual.storage_group,
            (i32::from(actual.tile_offset) - i32::from(template.tile_offset))
                .rem_euclid(i32::from(tile_count)),
        );
        if group.is_some_and(|old| old != binding) {
            return None; // Bound members of one ownership group must agree.
        }
        *group = Some(binding);
        ids[from.index() as usize] = Some(to);
    }
    let anchor = fragment
        .outputs
        .first()
        .copied()
        .zip(outputs.first().copied())
        .or_else(|| {
            fragment
                .inputs
                .first()
                .map(|v| v.value)
                .zip(inputs.first().copied())
        });
    let (origin, rotation) = anchor.map_or((None, 0), |(from, to)| {
        let actual = &values[to.index() as usize];
        (
            Some(actual.origin),
            i32::from(actual.tile_offset)
                - i32::from(fragment.values[from.index() as usize].tile_offset),
        )
    });
    let mut next = values.len();
    for value in &fragment.values {
        let binding = &mut ids[value.id.index() as usize];
        if binding.is_none() {
            let id = MidValueId::from_index(u32::try_from(next).ok()?);
            next = next.checked_add(1)?;
            *binding = Some(id);
            groups[value.storage_group.index() as usize].get_or_insert((id, rotation));
        }
    }
    let mut added = Vec::with_capacity(next - values.len());
    for template in &fragment.values {
        let id = ids[template.id.index() as usize]?;
        if id.index() as usize >= values.len() {
            let (storage_group, rotation) = groups[template.storage_group.index() as usize]?;
            added.push(MidValue {
                id,
                storage_group,
                tile_offset: (i32::from(template.tile_offset) + rotation)
                    .rem_euclid(i32::from(tile_count)) as u16,
                tensor_type: template.tensor_type.clone(),
                origin: origin.unwrap_or(template.origin),
            });
        }
    }
    let mut bound = fragment.operations.clone();
    remap_operations(&mut bound, &ids, source)?;
    bound.extend(returned.into_iter().map(|(input, result)| MidOperation {
        source,
        inputs: vec![input],
        results: vec![result],
        kind: MidOperationKind::Copy {
            mapping: CoordinateMapping::default(),
            reuse_local: true,
            policy: CopyPolicy::Automatic,
            packing: PackingPolicy::Automatic,
        },
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
    }));
    values.extend(added);
    operations.extend(bound);
    Some(())
}

fn remap_operations(
    operations: &mut [MidOperation],
    ids: &[Option<MidValueId>],
    source: Option<OperationId>,
) -> Option<()> {
    let remap = |value: &mut MidValueId| {
        *value = ids.get(value.index() as usize).copied().flatten()?;
        Some(())
    };
    for operation in operations {
        operation.source = source.or(operation.source);
        for value in operation.inputs.iter_mut().chain(&mut operation.results) {
            remap(value)?;
        }
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            for value in repeat
                .iterated_inputs
                .iter_mut()
                .flatten()
                .chain(&mut repeat.body.arguments)
                .chain(&mut repeat.body.yields)
            {
                remap(value)?;
            }
            remap_operations(&mut repeat.body.operations, ids, source)?;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphInputKind, ValueId};
    use crate::kernel::TileKernelSpec;
    use crate::mid::{Compute, MidInput, MidRegion, MidRepeat, OperandIndexing};
    use crate::tensor::{Layout, Precision, TensorType};

    fn id(i: u32) -> MidValueId {
        MidValueId::from_index(i)
    }
    fn value(i: u32) -> MidValue {
        MidValue {
            id: id(i),
            tile_offset: 0,
            tensor_type: TensorType::new([1, 16], Precision::F16, Layout::row_sharded(1)),
            origin: ValueId::from_index(i),
            storage_group: id(i),
        }
    }
    fn input(i: u32) -> MidInput {
        MidInput {
            name: format!("input{i}"),
            kind: GraphInputKind::Host,
            value: id(i),
        }
    }
    fn repeated_add() -> MidProgram {
        let mut values = (0..7).map(value).collect::<Vec<_>>();
        for i in [0, 3, 4, 6] {
            values[i].storage_group = id(0);
            values[i].tile_offset = 1;
        }
        for i in [1, 2, 5] {
            values[i].storage_group = id(1);
        }
        let add = MidOperation {
            source: None,
            inputs: vec![id(4), id(5)],
            results: vec![id(6)],
            kind: MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::Add,
                operands: vec![OperandIndexing::Elementwise { result: 0 }; 2],
                output_aliases: vec![(0, 0)],
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        MidProgram {
            tile_count: 4,
            values,
            inputs: (0..3).map(input).collect(),
            outputs: vec![id(3)],
            operations: vec![MidOperation {
                source: None,
                inputs: vec![id(0)],
                results: vec![id(3)],
                kind: MidOperationKind::Repeat(MidRepeat {
                    count: 2,
                    carried_inputs: 1,
                    invariant_inputs: 0,
                    iterated_inputs: vec![vec![id(1), id(2)]],
                    body: MidRegion {
                        arguments: vec![id(4), id(5)],
                        operations: vec![add],
                        yields: vec![id(6)],
                        estimated_cycles: 0,
                        peak_memory: Default::default(),
                    },
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            }],
            ..MidProgram::default()
        }
    }
    fn caller() -> MidProgram {
        let mut values = (0..6).map(value).collect::<Vec<_>>();
        for i in [2, 5] {
            values[i].storage_group = id(2);
            values[i].tile_offset = 7;
        }
        for i in [3, 4] {
            values[i].storage_group = id(3);
            values[i].tile_offset = 9;
        }
        MidProgram {
            tile_count: 16,
            inputs: (0..5).map(input).collect(),
            outputs: vec![id(5)],
            values,
            ..MidProgram::default()
        }
    }

    // Scalar execution suffices for these elementwise fixtures and verifies the
    // region bindings independently of the compiler's remapping implementation.
    fn execute(operations: &[MidOperation], values: &mut [i64]) {
        for op in operations {
            let result = match &op.kind {
                MidOperationKind::Copy { .. } => vec![values[op.inputs[0].index() as usize]],
                MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::Add,
                    ..
                }) => vec![op.inputs.iter().map(|i| values[i.index() as usize]).sum()],
                MidOperationKind::Repeat(repeat) => {
                    let mut arguments = op
                        .inputs
                        .iter()
                        .map(|i| values[i.index() as usize])
                        .collect::<Vec<_>>();
                    for iteration in 0..repeat.count as usize {
                        let iterated = repeat
                            .iterated_inputs
                            .iter()
                            .map(|sequence| values[sequence[iteration].index() as usize])
                            .collect::<Vec<_>>();
                        for (&id, value) in repeat
                            .body
                            .arguments
                            .iter()
                            .zip(arguments.iter().chain(&iterated))
                        {
                            values[id.index() as usize] = *value;
                        }
                        execute(&repeat.body.operations, values);
                        for (argument, &yielded) in arguments.iter_mut().zip(&repeat.body.yields) {
                            *argument = values[yielded.index() as usize];
                        }
                    }
                    arguments[..repeat.carried_inputs].to_vec()
                }
                other => panic!("unexpected fixture operation {other:?}"),
            };
            for (&id, value) in op.results.iter().zip(result) {
                values[id.index() as usize] = value;
            }
        }
    }

    #[test]
    fn binding_preserves_repeat_execution_ownership_and_aliases() {
        let fragment = repeated_add();
        let mut bound = caller();
        fragment.validate().unwrap();
        append_fragment(
            &fragment,
            &[id(2), id(3), id(4)],
            &[id(5)],
            None,
            bound.tile_count,
            &mut bound.values,
            &mut bound.operations,
        )
        .unwrap();
        bound.validate().unwrap();
        let mut before = vec![0; fragment.values.len()];
        before[..3].copy_from_slice(&[3, 5, 7]);
        let mut after = vec![0; bound.values.len()];
        after[2..5].copy_from_slice(&[3, 5, 7]);
        execute(&fragment.operations, &mut before);
        execute(&bound.operations, &mut after);
        assert_eq!(after[5], before[3]);
        assert_eq!(after[5], 15);
        let MidOperationKind::Repeat(repeat) = &bound.operations[0].kind else {
            panic!()
        };
        for &i in [&repeat.body.arguments[..1], &repeat.body.yields]
            .concat()
            .iter()
        {
            let value = &bound.values[i.index() as usize];
            assert_eq!((value.storage_group, value.tile_offset), (id(2), 7));
        }
        let weight = &bound.values[repeat.body.arguments[1].index() as usize];
        assert_eq!((weight.storage_group, weight.tile_offset), (id(3), 9));
        let MidOperationKind::Compute(compute) = &repeat.body.operations[0].kind else {
            panic!()
        };
        assert_eq!(compute.output_aliases(), &[(0, 0)]);
    }

    #[test]
    fn incompatible_or_invalid_bindings_fail_without_mutation() {
        for variant in 0..4 {
            let mut fragment = repeated_add();
            let mut bound = caller();
            match variant {
                0 => bound.values[5].tensor_type.format.precision = Precision::F32,
                1 => bound.values[5].storage_group = id(5),
                2 => {
                    let MidOperationKind::Repeat(repeat) = &mut fragment.operations[0].kind else {
                        panic!()
                    };
                    repeat.body.operations[0].inputs[0] = id(999);
                }
                3 => bound.values[2].tensor_type.shape.0[0] = 2,
                _ => unreachable!(),
            }
            let before = bound.clone();
            assert!(
                append_fragment(
                    &fragment,
                    &[id(2), id(3), id(4)],
                    &[id(5)],
                    None,
                    bound.tile_count,
                    &mut bound.values,
                    &mut bound.operations
                )
                .is_none()
            );
            assert_eq!(bound, before);
        }
    }

    #[test]
    fn returning_an_input_defines_every_requested_result() {
        let fragment = MidProgram {
            tile_count: 1,
            inputs: vec![input(0)],
            values: vec![value(0)],
            outputs: vec![id(0), id(0)],
            ..MidProgram::default()
        };
        let mut bound = MidProgram {
            tile_count: 1,
            inputs: vec![input(1)],
            values: (0..4).map(value).collect(),
            outputs: vec![id(2), id(3)],
            ..MidProgram::default()
        };
        append_fragment(
            &fragment,
            &[id(1)],
            &[id(2), id(3)],
            None,
            1,
            &mut bound.values,
            &mut bound.operations,
        )
        .unwrap();
        bound.validate().unwrap();
        let mut values = vec![0, 11, 0, 0];
        execute(&bound.operations, &mut values);
        assert_eq!(&values[2..], &[11, 11]);
        assert_eq!(bound.values[1].storage_group, id(1));
    }
}
