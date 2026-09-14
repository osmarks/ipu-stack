//! Compose consecutive materializations before tile expansion.

use super::*;

/// Batch adjacent independent copies. Their local preparations precede one
/// shared exchange. Diagnostic builds retain semantic checkpoint boundaries.
pub(crate) fn independent_copy_prefix(
    operations: &[MidOperation],
    checkpoints: bool,
    storage_groups: &[MidValueId],
) -> usize {
    independent_prefix(operations, checkpoints, storage_groups, |kind| {
        matches!(kind, MidOperationKind::Primitive(Primitive::Copy { .. }))
    })
}

pub(crate) fn independent_sum_prefix(
    operations: &[MidOperation],
    checkpoints: bool,
    storage_groups: &[MidValueId],
) -> usize {
    independent_prefix(operations, checkpoints, storage_groups, |kind| {
        matches!(kind, MidOperationKind::Primitive(Primitive::Sum { .. }))
    })
}

fn independent_prefix(
    operations: &[MidOperation],
    checkpoints: bool,
    storage_groups: &[MidValueId],
    eligible: impl Fn(&MidOperationKind) -> bool,
) -> usize {
    let group = |id: &MidValueId| storage_groups[id.index() as usize];
    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let source = operations.first().and_then(|operation| operation.source);
    operations
        .iter()
        .take_while(|operation| {
            if (checkpoints && operation.source != source)
                || !eligible(&operation.kind)
                || operation
                    .inputs
                    .iter()
                    .any(|id| outputs.contains(&group(id)))
                || operation
                    .results
                    .iter()
                    .any(|id| inputs.contains(&group(id)) || outputs.contains(&group(id)))
            {
                return false;
            }
            inputs.extend(operation.inputs.iter().map(group));
            outputs.extend(operation.results.iter().map(group));
            true
        })
        .count()
}

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

pub(super) fn compose_region(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) {
    for op in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut op.kind {
            compose_region(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    compose(operations, values, required);
}

pub(super) fn compose(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) {
    let mut uses = vec![0usize; values.len()];
    for input in operations
        .iter()
        .flat_map(MidOperation::read_values)
        .chain(required)
    {
        uses[input.index() as usize] += 1;
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
            // Keep pack-once staging when the first conversion needs local
            // packing. Native-panel copies can multicast straight to consumers;
            // retaining an intermediate merely adds a receive/forward hop.
            let source_format = &values[source.index() as usize].tensor_type.format;
            let exchange_only = previous.view.is_none()
                && (source_format.layout.order == intermediate.format.layout.order
                    || source_format.supports_micro_panel_exchange(&intermediate.format));
            if destination.format.layout.tiling.replicas
                > intermediate.format.layout.tiling.replicas
                && !exchange_only
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
        }
        if input != operations[index].inputs[0] {
            operations[index].inputs[0] = input;
            operations[index].kind = MidOperationKind::Primitive(Primitive::Copy {
                mapping: next,
                reuse_local,
            });
            operations[index].estimated_cycles = 0;
            operations[index].estimated_exchange_cycles = 0;
        }
        producers.insert(output, index);
    }
    super::rewrite::apply_edits(operations, &removed, BTreeMap::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_identity_copies_do_not_pay_for_an_exchange() {
        let (operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.format.layout = Layout::logical_linear(3, 4);
        }
        let (price, _, _) = crate::estimate::operation_cost(&operations[0], &values).unwrap();
        assert_eq!(price.exchange, 0);
        assert!(super::super::rewrite::same_storage(&values[0], &values[1]));

        values[1].tensor_type.shape.0[0] += 1;
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));
        values[1].tensor_type = values[0].tensor_type.clone();
        values[1].tile_offset = 1;
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));

        for value in &mut values {
            value.tile_offset = 0;
            value.tensor_type.format.layout = Layout::row_sharded(1);
        }
        values[1].tensor_type.format.layout.tiling.axes[0].shard_padding_multiple = 8;
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));
    }

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
    fn native_fp8_panel_materialization_composes_into_replication() {
        let (mut operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.shape = TensorShape(vec![64, 16]);
            value.tensor_type.format.precision = crate::Precision::F8F143 { scale_exponent: -4 };
            value.tensor_type.format.layout.order =
                crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: 16,
                });
        }
        values[0].tensor_type.format.layout.order =
            crate::ElementOrder::Amp(crate::AmpOrder::TransposedLeft);
        values[2].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        values[3].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);
        assert_eq!(operations[0].results, [MidValueId(3)]);
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
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);

        let (mut operations, mut values) = chain();
        values[0].tensor_type.format.layout.order =
            crate::ElementOrder::Amp(crate::AmpOrder::Output);
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
