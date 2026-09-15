//! Bind an already executable fragment into an enclosing mid region.
use super::{CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId};
use crate::graph::OperationId;
use crate::low::{CopyPolicy, PackingPolicy};
use std::collections::BTreeMap;

/// Substitute complete boundary bindings, preserving the fragment's operations
/// and ownership groups. Unbound groups use the explicit working embedding
/// unless the fragment declares one, preserving relative rotations. A small
/// result's embedding does not restrict the fragment's intermediate work.
/// Groups shared with a boundary retain that boundary's mapping instead.
/// New temporaries belong to the caller's semantic result for diagnostics.
/// A returned input is connected to a distinct caller result by an explicit
/// identity copy, which ordinary copy lowering can reuse locally.
/// Failure leaves both caller vectors unchanged.
pub(crate) fn append_fragment(
    fragment: &MidProgram,
    inputs: &[MidValueId],
    outputs: &[MidValueId],
    working: &crate::tensor::OwnerMap,
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
            || actual.storage_group.index() as usize >= values.len()
        {
            return None;
        }
        actual
            .owners
            .validate(
                actual.tensor_type.format.layout.tiling.tile_count,
                tile_count,
            )
            .ok()?;
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
            actual
                .owners
                .shifted(-i32::from(template.owners.rotation()), tile_count)?,
        );
        if group.as_ref().is_some_and(|old| old != &binding) {
            return None; // Bound members of one ownership group must agree.
        }
        *group = Some(binding);
        ids[from.index() as usize] = Some(to);
    }
    let origin = outputs
        .first()
        .or_else(|| inputs.first())
        .map(|to| values[to.index() as usize].origin);
    let mut next = values.len();
    for value in &fragment.values {
        let binding = &mut ids[value.id.index() as usize];
        if binding.is_none() {
            let id = MidValueId::from_index(u32::try_from(next).ok()?);
            next = next.checked_add(1)?;
            *binding = Some(id);
            let owners = if value.owners.has_embedding() {
                value.owners.with_rotation(working.rotation())
            } else {
                working.clone()
            };
            groups[value.storage_group.index() as usize].get_or_insert((id, owners));
        }
    }
    let mut added = Vec::with_capacity(next - values.len());
    for template in &fragment.values {
        let id = ids[template.id.index() as usize]?;
        if id.index() as usize >= values.len() {
            let (storage_group, owners) =
                groups[template.storage_group.index() as usize].as_ref()?;
            let owners = owners.shifted(i32::from(template.owners.rotation()), tile_count)?;
            owners
                .validate(
                    template.tensor_type.format.layout.tiling.tile_count,
                    tile_count,
                )
                .ok()?;
            added.push(MidValue {
                id,
                storage_group: *storage_group,
                owners,
                tensor_type: template.tensor_type.clone(),
                origin: origin.unwrap_or(template.origin),
            });
        }
    }
    let mut bound = fragment.operations.clone();
    remap_operations(&mut bound, &ids, source)?;
    bound.extend(returned.into_iter().map(|(input, result)| MidOperation {
        site: None,
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
            owners: crate::tensor::OwnerMap::default(),
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
            values[i].owners = crate::tensor::OwnerMap::rotated(1);
        }
        for i in [1, 2, 5] {
            values[i].storage_group = id(1);
        }
        let add = MidOperation {
            site: None,
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
        let mut program = MidProgram {
            tile_count: 4,
            values,
            inputs: (0..3).map(input).collect(),
            outputs: vec![id(3)],
            operations: vec![MidOperation {
                site: None,
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
        };
        crate::mid::ownership::bind_owners(&mut program.operations, &mut program.values).unwrap();
        program
    }
    fn caller() -> MidProgram {
        let mut values = (0..6).map(value).collect::<Vec<_>>();
        for i in [2, 5] {
            values[i].storage_group = id(2);
            values[i].owners = crate::tensor::OwnerMap::rotated(7);
        }
        for i in [3, 4] {
            values[i].storage_group = id(3);
            values[i].owners = crate::tensor::OwnerMap::rotated(9);
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
    fn repeat_copies_a_moved_body_result_back_to_its_carried_home() {
        let mut program = repeated_add();
        let mut graph = crate::ComputeGraph::new();
        let input = graph.host_input("x", [1, 16]).unwrap();
        graph.add(input, input).unwrap();
        let source = graph.operations()[0].id;
        let MidOperationKind::Repeat(repeat) = &mut program.operations[0].kind else {
            unreachable!()
        };
        let add = repeat.body.operations.last_mut().unwrap();
        add.source = Some(source);
        add.site = Some("add".into());
        let site = add.result_site(0).unwrap();
        let home = program.values[add.results[0].index() as usize]
            .owners
            .clone();
        let mut choices = crate::mid::OwnerChoices::default();
        choices.results.insert(site.clone(), home);
        let original = program.clone();
        program.apply_ownership(&choices).unwrap();
        assert_eq!(
            program, original,
            "an explicit current assignment preserves the input's home"
        );
        choices
            .results
            .insert(site, crate::tensor::OwnerMap::embedded(vec![3]));
        assert!(program.apply_ownership(&choices).is_err());
        assert_eq!(
            program, original,
            "a result choice cannot move aliased input storage"
        );
        let MidOperationKind::Repeat(repeat) = &mut program.operations[0].kind else {
            unreachable!()
        };
        let add = repeat.body.operations.last_mut().unwrap();
        let MidOperationKind::Compute(Compute::Kernel { output_aliases, .. }) = &mut add.kind
        else {
            unreachable!()
        };
        output_aliases.clear();
        program.values[6].storage_group = id(6);
        program.values[6].owners = crate::tensor::OwnerMap::default();
        let inputs = program.values[..3].to_vec();
        program.apply_ownership(&choices).unwrap();
        program.validate().unwrap();
        crate::low::expand::expand_tiles(&program, false).unwrap();
        assert_eq!(program.values[..3], inputs);
        let MidOperationKind::Repeat(repeat) = &program.operations[0].kind else {
            unreachable!()
        };
        let yielded = &program.values[repeat.body.yields[0].index() as usize];
        assert_eq!(yielded.owners, program.values[0].owners);
        assert_ne!(yielded.owners, program.values[6].owners);
        let mut values = vec![0; program.values.len()];
        values[..3].copy_from_slice(&[3, 5, 7]);
        execute(&program.operations, &mut values);
        assert_eq!(values[3], 15);
    }

    #[test]
    fn small_result_home_does_not_restrict_working_owners() {
        let mut fragment = MidProgram {
            tile_count: 4,
            inputs: vec![input(0)],
            outputs: vec![id(2)],
            values: (0..3).map(value).collect(),
            ..MidProgram::default()
        };
        for index in 0..3 {
            fragment.values[index].tensor_type = TensorType::new(
                [4, 16],
                Precision::F16,
                Layout::row_sharded(if index == 2 { 1 } else { 4 }),
            );
        }
        for index in 0..2 {
            fragment.operations.push(MidOperation {
                source: None,
                site: None,
                inputs: vec![id(index)],
                results: vec![id(index + 1)],
                kind: MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                    policy: CopyPolicy::Automatic,
                    packing: PackingPolicy::Automatic,
                },
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            });
        }
        let mut result = fragment.values[2].clone();
        result.id = id(1);
        result.storage_group = id(1);
        result.owners = crate::tensor::OwnerMap::embedded(vec![10]);
        let mut bound = MidProgram {
            tile_count: 16,
            inputs: vec![input(0)],
            outputs: vec![id(1)],
            values: vec![fragment.values[0].clone(), result],
            ..MidProgram::default()
        };
        bound.values[0].owners = crate::tensor::OwnerMap::embedded(vec![1, 4, 7, 9]);
        let original = bound.clone();
        let working = crate::tensor::OwnerMap::embedded(vec![2, 3, 5, 6]);
        append_fragment(
            &fragment,
            &[id(0)],
            &[id(1)],
            &working,
            None,
            bound.tile_count,
            &mut bound.values,
            &mut bound.operations,
        )
        .unwrap();
        assert_eq!(bound.values[0..2], original.values);
        assert_eq!(bound.values[2].owners, working);
        bound.validate().unwrap();
        crate::low::expand::expand_tiles(&bound, false).unwrap();

        let mut invalid = original.clone();
        assert!(
            append_fragment(
                &fragment,
                &[id(0)],
                &[id(1)],
                &crate::tensor::OwnerMap::embedded(vec![2]),
                None,
                invalid.tile_count,
                &mut invalid.values,
                &mut invalid.operations
            )
            .is_none()
        );
        assert_eq!(invalid, original);
    }

    #[test]
    fn binding_preserves_repeat_execution_ownership_and_aliases() {
        for embedded in [false, true] {
            let fragment = repeated_add();
            let mut bound = caller();
            if embedded {
                for i in [2, 5] {
                    bound.values[i].owners =
                        crate::tensor::OwnerMap::embedded(vec![2, 7, 10, 3]).with_rotation(1);
                }
                for i in [3, 4] {
                    bound.values[i].owners =
                        crate::tensor::OwnerMap::embedded(vec![11, 5, 9]).with_rotation(2);
                }
            }
            fragment.validate().unwrap();
            let working = bound.values[5]
                .owners
                .shifted(-1, bound.tile_count)
                .unwrap();
            append_fragment(
                &fragment,
                &[id(2), id(3), id(4)],
                &[id(5)],
                &working,
                None,
                bound.tile_count,
                &mut bound.values,
                &mut bound.operations,
            )
            .unwrap();
            bound.validate().unwrap();
            crate::low::expand::expand_tiles(&bound, false).unwrap();
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
                assert_eq!(value.storage_group, id(2));
                assert_eq!(value.owners, bound.values[2].owners);
            }
            let weight = &bound.values[repeat.body.arguments[1].index() as usize];
            assert_eq!(weight.storage_group, id(3));
            assert_eq!(weight.owners, bound.values[3].owners);
            let MidOperationKind::Compute(compute) = &repeat.body.operations.last().unwrap().kind
            else {
                panic!()
            };
            assert_eq!(compute.output_aliases(), &[(0, 0)]);
        }
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
                    &crate::tensor::OwnerMap::default(),
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
            &crate::tensor::OwnerMap::default(),
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
