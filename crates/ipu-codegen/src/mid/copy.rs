//! Compose consecutive materializations before tile expansion.

use super::*;

fn mapping(operation: &MidOperation) -> Option<(CoordinateMapping, bool)> {
    match &operation.kind {
        MidOperationKind::Primitive(Primitive::Copy {
            mapping,
            reuse_local,
        }) => Some((mapping.clone(), *reuse_local)),
        MidOperationKind::Convert(_) => Some((CoordinateMapping::default(), false)),
        _ => None,
    }
}

pub(super) fn compose(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) {
    let mut uses = vec![0usize; values.len()];
    for input in operations
        .iter()
        .flat_map(|operation| &operation.inputs)
        .chain(required)
    {
        uses[input.index() as usize] += 1;
    }
    for operation in operations.iter() {
        if let MidOperationKind::Repeat(repeat) = &operation.kind {
            for input in repeat.iterated_inputs.iter().flatten() {
                uses[input.index() as usize] += 1;
            }
        }
    }
    let mut producers = BTreeMap::<MidValueId, usize>::new();
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let Some((mut next, mut reuse_local)) = mapping(&operations[index]) else {
            // In-place compute and loop-carried storage can overwrite an
            // earlier source. Do not move a materialization across them.
            producers.clear();
            continue;
        };
        let ([input], [output]) = (
            operations[index].inputs.as_slice(),
            operations[index].results.as_slice(),
        ) else {
            continue;
        };
        let output = *output;
        let mut input = *input;
        while uses[input.index() as usize] == 1 {
            let Some(&producer) = producers.get(&input) else {
                break;
            };
            let (previous, previous_reuse) = mapping(&operations[producer]).unwrap();
            let source = operations[producer].inputs[0];
            let intermediate = &values[input.index() as usize].tensor_type;
            let destination = &values[output.index() as usize].tensor_type;
            // Byte movement cannot itself convert element widths. Round trips
            // may disappear, but a remaining precision change needs its Convert.
            if values[source.index() as usize].tensor_type.format.precision
                != destination.format.precision
            {
                break;
            }
            // Keep pack-once/broadcast staging. Folding a local rearrangement
            // into replication repeats that work on every receiving tile.
            if destination.format.layout.tiling.replicas
                > intermediate.format.layout.tiling.replicas
            {
                break;
            }
            let Some(composed) = previous.compose(
                &next,
                &values[source.index() as usize].tensor_type.shape,
                &intermediate.shape,
                &destination.shape,
            ) else {
                break;
            };
            next = composed;
            reuse_local &= previous_reuse;
            input = source;
            removed.insert(producer);
            operations[index].inputs[0] = input;
            operations[index].kind = MidOperationKind::Primitive(Primitive::Copy {
                mapping: next.clone(),
                reuse_local,
            });
            operations[index].estimated_cycles = 0;
            operations[index].estimated_exchange_cycles = 0;
        }
        producers.insert(output, index);
    }
    let mut index = 0;
    operations.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate(
        mapping: &CoordinateMapping,
        source: &TensorShape,
        mut point: Vec<u32>,
    ) -> Vec<u32> {
        for (axis, offset) in mapping.offsets.iter().enumerate() {
            point[axis] += offset;
        }
        if let Some(view) = mapping.view {
            let width = source.0[view.split_axis] / view.factor;
            point[view.split_axis] += point[view.merge_axis] % view.factor * width;
            point[view.merge_axis] /= view.factor;
        }
        point
    }

    #[test]
    fn composed_windows_and_factor_views_preserve_coordinates() {
        let mut rng = fastrand::Rng::with_seed(0x636f6d706f7365);
        let mut accepted = [0; 4];
        for _ in 0..4000 {
            let source = TensorShape(vec![12, 12, 12]);
            let mut make = |shape: &TensorShape| {
                let split = rng.usize(0..3);
                let merge = (split + rng.usize(1..3)) % 3;
                let view = rng.bool().then(|| AxisFactorView::new(split, merge, 2));
                let mut output =
                    view.map_or_else(|| Some(shape.clone()), |view| view.output_shape(shape))?;
                let offsets = output
                    .0
                    .iter_mut()
                    .map(|size| {
                        let offset = rng.u32(0..=(*size).min(1));
                        *size -= offset;
                        offset
                    })
                    .collect();
                Some((CoordinateMapping { offsets, view }, output))
            };
            let Some((first, middle)) = make(&source) else {
                continue;
            };
            let Some((second, output)) = make(&middle) else {
                continue;
            };
            let Some(composed) = first.compose(&second, &source, &middle, &output) else {
                continue;
            };
            accepted[usize::from(first.view.is_some()) * 2 + usize::from(second.view.is_some())] +=
                1;
            for _ in 0..10 {
                let point = output
                    .0
                    .iter()
                    .map(|&size| rng.u32(0..size))
                    .collect::<Vec<_>>();
                assert_eq!(
                    coordinate(&composed, &source, point.clone()),
                    coordinate(&first, &source, coordinate(&second, &middle, point))
                );
            }
        }
        assert!(accepted.iter().all(|&count| count > 0), "{accepted:?}");
    }

    fn chain() -> (Vec<MidOperation>, Vec<MidValue>) {
        let values = (0..4)
            .map(|index| MidValue {
                id: MidValueId(index),
                tile_offset: 0,
                tensor_type: TensorType {
                    shape: TensorShape(vec![4, 16]),
                    format: TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::row_major(TensorTiling::replicated(1)),
                    },
                },
                origin: ValueId::from_index(0),
                storage_group: MidValueId(index),
            })
            .collect();
        let operations = (1..4)
            .map(|index| MidOperation {
                source: None,
                inputs: vec![MidValueId(index - 1)],
                results: vec![MidValueId(index)],
                kind: MidOperationKind::Primitive(Primitive::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: true,
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            })
            .collect();
        (operations, values)
    }

    #[test]
    fn copy_chains_drop_temporaries_and_intermediate_rounding() {
        let (mut operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.format.precision = Precision::F32;
        }
        values[1].tensor_type.format.precision = Precision::F16;
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);
        assert_eq!(operations[0].results, [MidValueId(3)]);
        let once = operations.clone();
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations, once);
    }

    #[test]
    fn composition_keeps_required_precision_changes_and_cropped_zeros() {
        let (mut operations, mut values) = chain();
        values[0].tensor_type.format.precision = Precision::F32;
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);

        let identity = CoordinateMapping::default();
        let view = CoordinateMapping::from(AxisFactorView::new(2, 0, 3));
        let source = TensorShape(vec![1, 2, 12]);
        let middle = TensorShape(vec![3, 2, 4]);
        let padded = TensorShape(vec![3, 2, 8]);
        assert_eq!(
            view.compose(&identity, &source, &middle, &padded),
            Some(view.clone())
        );
        let cropped = TensorShape(vec![3, 2, 2]);
        assert!(
            view.compose(&identity, &source, &cropped, &padded)
                .is_none()
        );
    }

    #[test]
    fn composition_preserves_shared_results_broadcast_staging_and_padding() {
        let (mut operations, values) = chain();
        compose(&mut operations, &values, &[MidValueId(1), MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);

        let (mut operations, mut values) = chain();
        values[2].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        values[3].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);

        let (mut operations, mut values) = chain();
        values[1].tensor_type.shape.0[1] += 4;
        values[2].tensor_type.shape.0[1] += 4;
        values[3].tensor_type.shape.0[1] += 4;
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);
    }
}
