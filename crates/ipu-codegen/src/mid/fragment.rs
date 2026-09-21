//! Bind executable work to enclosing inputs and return its actual result values.
use super::{MidGraph, MidOperation, MidValue, MidValueId};
use crate::graph::{OperationId, ValueId};
use crate::mid::MidOperationKind;
use crate::tensor::OwnerMap;

/// Inputs keep their existing ownership groups; new groups use the working
/// embedding and the fragment's relative rotations. Result homes are chosen by
/// the ordinary ownership pass, rather than by preallocating boundary slots.
/// Returning an input or returning one value twice needs no extra operation.
/// Failure leaves both caller vectors unchanged.
/// A missing origin override preserves per-value origins in multi-operation fragments.
pub(crate) fn append_fragment(
    fragment: &MidGraph,
    inputs: &[MidValueId],
    working: &OwnerMap,
    source: Option<OperationId>,
    origin: Option<ValueId>,
    tile_count: u16,
    values: &mut Vec<MidValue>,
    operations: &mut Vec<MidOperation>,
) -> Option<Vec<MidValueId>> {
    if tile_count == 0 || fragment.tile_count > tile_count || fragment.inputs.len() != inputs.len()
    {
        return None;
    }
    fragment.validate().ok()?;
    let mut ids = vec![None; fragment.values.len()];
    let mut groups = vec![None; fragment.values.len()];
    for (input, &to) in fragment.inputs.iter().zip(inputs) {
        let template = &fragment.values[input.value.index() as usize];
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
        let binding = (
            actual.storage_group,
            actual
                .owners
                .shifted(-i32::from(template.owners.rotation()), tile_count)?,
        );
        let group = &mut groups[template.storage_group.index() as usize];
        if group.as_ref().is_some_and(|old| old != &binding) {
            return None;
        }
        *group = Some(binding);
        ids[input.value.index() as usize] = Some(to);
    }
    let mut added = Vec::new();
    // Allocate results first to retain the existing stable allocation order.
    for &from in fragment
        .outputs
        .iter()
        .chain(fragment.values.iter().map(|v| &v.id))
    {
        if ids[from.index() as usize].is_none() {
            let id = MidValueId::from_index(u32::try_from(values.len() + added.len()).ok()?);
            ids[from.index() as usize] = Some(id);
            added.push(fragment.values[from.index() as usize].clone());
        }
    }
    let ids = ids.into_iter().collect::<Option<Vec<_>>>()?;
    for value in &added {
        let owners = if value.owners.has_embedding() {
            value.owners.with_rotation(working.rotation())
        } else {
            working.clone()
        };
        groups[value.storage_group.index() as usize]
            .get_or_insert((ids[value.id.index() as usize], owners));
    }
    for value in &mut added {
        let (storage_group, owners) = groups[value.storage_group.index() as usize].as_ref()?;
        value.owners = owners.shifted(i32::from(value.owners.rotation()), tile_count)?;
        value
            .owners
            .validate(
                value.tensor_type.format.layout.tiling.tile_count,
                tile_count,
            )
            .ok()?;
        value.id = ids[value.id.index() as usize];
        value.storage_group = *storage_group;
        if let Some(origin) = origin {
            value.origin = origin;
        }
    }
    let mut bound = fragment.operations.clone();
    remap_operations(&mut bound, &ids, source);
    let results = fragment
        .outputs
        .iter()
        .map(|id| ids[id.index() as usize])
        .collect();
    values.extend(added);
    operations.extend(bound);
    Some(results)
}

fn remap_operations(
    operations: &mut [MidOperation],
    ids: &[MidValueId],
    source: Option<OperationId>,
) {
    let remap = |value: &mut MidValueId| *value = ids[value.index() as usize];
    for operation in operations {
        operation.source = source.or(operation.source);
        for value in operation.inputs.iter_mut().chain(&mut operation.results) {
            remap(value);
        }
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            for value in repeat
                .iterated_inputs
                .iter_mut()
                .flatten()
                .chain(&mut repeat.body.arguments)
                .chain(&mut repeat.body.yields)
            {
                remap(value);
            }
            remap_operations(&mut repeat.body.operations, ids, source);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphInputKind, ValueId};

    use crate::mid::{CoordinateMapping, MidInput, MidRegion, MidRepeat, OperandIndexing};
    use crate::tensor::{Layout, Precision, TensorType};
    use crate::{CopyPolicy, PackingPolicy};

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
    fn repeated_add() -> MidGraph {
        let mut values = (0..7).map(value).collect::<Vec<_>>();
        for i in [0, 3, 4, 6] {
            values[i].storage_group = id(0);
            values[i].owners = crate::tensor::OwnerMap::rotated(1);
        }
        for i in [1, 2, 5] {
            values[i].storage_group = id(1);
        }
        let add = MidOperation {
            source: None,
            inputs: vec![id(4), id(5)],
            results: vec![id(6)],
            kind: MidOperationKind::Add,
            operands: vec![OperandIndexing::Elementwise { result: 0 }; 2],
            output_aliases: vec![(0, 0)],
            output_windows: Vec::new(),
        };
        let mut program = MidGraph {
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
                    },
                }),
                operands: Vec::new(),
                output_aliases: Vec::new(),
                output_windows: Vec::new(),
            }],
            ..MidGraph::default()
        };
        crate::mid::ownership::bind_owners(&mut program.operations, &mut program.values).unwrap();
        program
    }
    fn caller() -> MidGraph {
        let mut values = (0..5).map(value).collect::<Vec<_>>();
        for i in [2] {
            values[i].storage_group = id(2);
            values[i].owners = crate::tensor::OwnerMap::rotated(7);
        }
        for i in [3, 4] {
            values[i].storage_group = id(3);
            values[i].owners = crate::tensor::OwnerMap::rotated(9);
        }
        MidGraph {
            tile_count: 16,
            inputs: (0..5).map(input).collect(),
            values,
            ..MidGraph::default()
        }
    }

    // Scalar execution suffices for these elementwise fixtures and verifies the
    // region bindings independently of the compiler's remapping implementation.
    fn execute(operations: &[MidOperation], values: &mut [i64]) {
        for op in operations {
            let result = match &op.kind {
                MidOperationKind::Copy { .. } => vec![values[op.inputs[0].index() as usize]],
                MidOperationKind::Add => {
                    vec![op.inputs.iter().map(|i| values[i.index() as usize]).sum()]
                }
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
        let mut graph = crate::HighGraph::new();
        let input = graph.host_input("x", [1, 16]).unwrap();
        graph.add(input, input).unwrap();
        let source = graph.operations()[0].id;
        let MidOperationKind::Repeat(repeat) = &mut program.operations[0].kind else {
            unreachable!()
        };
        let add = repeat.body.operations.last_mut().unwrap();
        add.source = Some(source);

        let MidOperationKind::Repeat(repeat) = &mut program.operations[0].kind else {
            unreachable!()
        };
        let add = repeat.body.operations.last_mut().unwrap();
        let output_aliases = &mut add.output_aliases;
        output_aliases.clear();
        program.values[6].storage_group = id(6);
        program.values[6].owners = crate::tensor::OwnerMap::embedded(vec![3]);
        let inputs = program.values[..3].to_vec();
        super::super::ownership::bind_owners(&mut program.operations, &mut program.values).unwrap();
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
        let mut graph = crate::HighGraph::new();
        let input_value = graph.host_input("x", [4, 16]).unwrap();
        graph.gelu(input_value).unwrap();
        let mut fragment = MidGraph {
            tile_count: 4,
            inputs: vec![input(0)],
            outputs: vec![id(2)],
            values: (0..3).map(value).collect(),
            ..MidGraph::default()
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
                inputs: vec![id(index)],
                results: vec![id(index + 1)],
                kind: MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    policy: CopyPolicy::Automatic,
                    packing: PackingPolicy::Staged,
                },
                operands: Vec::new(),
                output_aliases: Vec::new(),
                output_windows: Vec::new(),
            });
        }
        let mut bound = MidGraph {
            tile_count: 16,
            inputs: vec![input(0)],
            values: vec![fragment.values[0].clone()],
            ..MidGraph::default()
        };
        bound.values[0].owners = OwnerMap::embedded(vec![1, 4, 7, 9]);
        let original = bound.clone();
        let working = OwnerMap::embedded(vec![2, 3, 5, 6]);
        bound.outputs = append_fragment(
            &fragment,
            &[id(0)],
            &working,
            Some(graph.operations()[0].id),
            Some(ValueId::from_index(0)),
            bound.tile_count,
            &mut bound.values,
            &mut bound.operations,
        )
        .unwrap();
        bound.values[bound.outputs[0].index() as usize].owners = OwnerMap::embedded(vec![10]);
        super::super::ownership::bind_owners(&mut bound.operations, &mut bound.values).unwrap();
        assert_eq!(bound.values[..1], original.values);
        assert_eq!(bound.values[2].owners, working);
        assert_eq!(
            bound.values[bound.outputs[0].index() as usize].owners,
            OwnerMap::embedded(vec![10])
        );
        bound.validate().unwrap();
        crate::low::expand::expand_tiles(&bound, false).unwrap();

        let mut invalid = original.clone();
        assert!(
            append_fragment(
                &fragment,
                &[id(0)],
                &crate::tensor::OwnerMap::embedded(vec![2]),
                None,
                Some(ValueId::from_index(0)),
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
                for i in [2] {
                    bound.values[i].owners =
                        crate::tensor::OwnerMap::embedded(vec![2, 7, 10, 3]).with_rotation(1);
                }
                for i in [3, 4] {
                    bound.values[i].owners =
                        crate::tensor::OwnerMap::embedded(vec![11, 5, 9]).with_rotation(2);
                }
            }
            fragment.validate().unwrap();
            let working = bound.values[2]
                .owners
                .shifted(-1, bound.tile_count)
                .unwrap();
            bound.outputs = append_fragment(
                &fragment,
                &[id(2), id(3), id(4)],
                &working,
                None,
                Some(ValueId::from_index(5)),
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
            let compute = repeat.body.operations.last().unwrap();
            assert_eq!(compute.output_aliases.as_slice(), &[(0, 0)]);
        }
    }

    #[test]
    fn incompatible_or_invalid_bindings_fail_without_mutation() {
        for variant in 0..4 {
            let mut fragment = repeated_add();
            let mut bound = caller();
            match variant {
                0 => bound.values[2].tensor_type.format.precision = Precision::F32,
                1 => bound.values[4].storage_group = id(4),
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
                    &crate::tensor::OwnerMap::default(),
                    None,
                    Some(ValueId::from_index(5)),
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
    fn returning_an_input_reuses_the_value_in_every_result_slot() {
        let fragment = MidGraph {
            tile_count: 1,
            inputs: vec![input(0)],
            values: vec![value(0)],
            outputs: vec![id(0), id(0)],
            ..MidGraph::default()
        };
        let mut bound = MidGraph {
            tile_count: 1,
            inputs: vec![input(1)],
            values: (0..2).map(value).collect(),
            ..MidGraph::default()
        };
        bound.outputs = append_fragment(
            &fragment,
            &[id(1)],
            &crate::tensor::OwnerMap::default(),
            None,
            Some(ValueId::from_index(0)),
            1,
            &mut bound.values,
            &mut bound.operations,
        )
        .unwrap();
        bound.validate().unwrap();
        let mut values = vec![0, 11];
        execute(&bound.operations, &mut values);
        assert_eq!(
            bound
                .outputs
                .iter()
                .map(|id| values[id.index() as usize])
                .collect::<Vec<_>>(),
            [11, 11]
        );
        assert_eq!(bound.values.len(), 2);
        assert_eq!(bound.values[1].storage_group, id(1));
    }
}
