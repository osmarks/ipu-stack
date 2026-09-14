//! Fuse supported producer epilogues before or after redistribution.
use super::rewrite::{apply_edits, producer_through_copies, same_storage};
use super::*;

// A producer may write a cast's explicit result when its F16 intermediate
// has no other readers. Keep the separate path whenever it costs less.
// Price both legal execution locations with the same matching/safety rules.
pub(super) fn fuse(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut best = None;
    for at_source in [false, true] {
        let mut ops = operations.clone();
        let mut vals = values.clone();
        if fuse_fp8_outputs_at(&mut ops, &mut vals, required, at_source)
            && let Some(cycles) = super::rewrite::operation_cycles(&ops, &vals)
            && best.as_ref().is_none_or(|(old, _, _)| cycles < *old)
        {
            best = Some((cycles, ops, vals));
        }
    }
    if let Some((_, ops, vals)) = best {
        *operations = ops;
        *values = vals;
        true
    } else {
        false
    }
}

fn fuse_fp8_outputs_at(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
    at_source: bool,
) -> bool {
    let mut removed = BTreeSet::new();
    let mut preparation = BTreeMap::<usize, Vec<MidOperation>>::new();
    for index in 0..operations.len() {
        let cast = operations[index].clone();
        if super::rewrite::fp8_cast(&cast, values).is_none() {
            continue;
        }
        let Some((intermediate, previous, identity_copies)) =
            producer_through_copies(cast.inputs[0], &operations[..index], values, false)
        else {
            continue;
        };
        if std::iter::once(intermediate)
            .chain(identity_copies.iter().map(|&i| operations[i].results[0]))
            .any(|value| {
                required.contains(&value)
                    || operations
                        .iter()
                        .filter(|op| op.read_values().any(|v| *v == value))
                        .count()
                        != 1
            })
        {
            continue;
        }
        if removed.contains(&previous) {
            continue;
        }
        let producer = operations[previous].clone();
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel,
            operands,
            product: None,
            ..
        }) = &producer.kind
        else {
            continue;
        };
        let Some(capability) = kernel.output_capability(
            values[cast.results[0].index() as usize]
                .tensor_type
                .format
                .precision,
        ) else {
            continue;
        };
        if producer.inputs.len() != capability.operands
            || operands.iter().any(|window| !window.0.is_empty())
        {
            continue;
        }
        let redistributed = !same_storage(
            &values[intermediate.index() as usize],
            &values[cast.inputs[0].index() as usize],
        );
        // Elementwise arithmetic commutes with coordinate-preserving distribution. Move its
        // input through the existing copies and run the fused producer on the
        // consumer's owners. LN additionally requires complete rows and affine copies.
        if redistributed
            && (producer.inputs.is_empty()
                || !same_storage(
                    &values[producer.inputs[0].index() as usize],
                    &values[intermediate.index() as usize],
                ))
        {
            continue;
        }
        let input = &values[if redistributed {
            cast.inputs[0]
        } else {
            intermediate
        }
        .index() as usize];
        let output = &values[cast.results[0].index() as usize];
        tracing::debug!(target: "ipu_codegen::mid::elementwise", producer = ?producer.source,
            consumer = ?cast.source, ?kernel, input_layout = ?input.tensor_type.format.layout,
            output_layout = ?output.tensor_type.format.layout, "considering direct FP8 output");
        if !(at_source && redistributed)
            && (input.tile_offset != output.tile_offset
                || input.tensor_type.format.layout.order != capability.input_order
                || !capability
                    .output_orders
                    .contains(&output.tensor_type.format.layout.order))
        {
            continue;
        }
        let expected = if output.tensor_type.format.layout.order == ElementOrder::RowMajor {
            Some(input.tensor_type.format.layout.clone())
        } else {
            input
                .tensor_type
                .fp8_producer_layout(&output.tensor_type.format)
        };
        if !(at_source && redistributed)
            && (expected.is_none_or(|layout| {
                !same_storage(
                    &MidValue {
                        tensor_type: TensorType {
                            shape: input.tensor_type.shape.clone(),
                            format: TensorFormat {
                                precision: output.tensor_type.format.precision,
                                layout,
                            },
                        },
                        ..input.clone()
                    },
                    output,
                )
            }) || input
                .tensor_type
                .format
                .layout
                .resolve(&input.tensor_type.shape)
                .ok()
                .and_then(|resolved| {
                    resolved
                        .axes()
                        .and_then(|a| a.last())
                        .map(|a| a.extents_are_multiple_of(capability.column_multiple))
                })
                != Some(true))
        {
            continue;
        }
        // Parameter copies may intervene, provided they cannot overwrite inputs.
        let groups = producer
            .inputs
            .iter()
            .map(|v| values[v.index() as usize].storage_group)
            .chain(
                identity_copies
                    .iter()
                    .map(|&i| values[operations[i].results[0].index() as usize].storage_group),
            )
            .collect::<BTreeSet<_>>();
        if operations[previous + 1..index]
            .iter()
            .enumerate()
            .any(|(i, op)| {
                !identity_copies.contains(&(previous + 1 + i))
                    && (matches!(op.kind, MidOperationKind::Repeat(_))
                        || op
                            .results
                            .iter()
                            .any(|v| groups.contains(&values[v.index() as usize].storage_group)))
            })
        {
            continue;
        }
        if at_source && redistributed {
            let source = &values[intermediate.index() as usize];
            if source.tensor_type.format.layout.order != capability.input_order {
                continue;
            }
            let Some(layout) = source
                .tensor_type
                .fp8_producer_layout(&output.tensor_type.format)
            else {
                continue;
            };
            if !capability.output_orders.contains(&layout.order)
                || !source
                    .tensor_type
                    .format
                    .layout
                    .resolve(&source.tensor_type.shape)
                    .ok()
                    .and_then(|r| {
                        r.axes()
                            .and_then(|a| a.last())
                            .map(|a| a.extents_are_multiple_of(capability.column_multiple))
                    })
                    .unwrap_or(false)
            {
                continue;
            }
            let mut value = source.clone();
            value.id = MidValueId(values.len() as u32);
            value.storage_group = value.id;
            value.tensor_type.format = TensorFormat {
                precision: output.tensor_type.format.precision,
                layout,
            };
            let mut fused = producer.clone();
            fused.results = vec![value.id];
            fused.kind = MidOperationKind::Primitive(Primitive::Compute {
                kernel: kernel.clone(),
                operands: operands.clone(),
                product: None,
                output_aliases: vec![],
            });
            let copy = MidOperation {
                inputs: vec![value.id],
                kind: MidOperationKind::Convert(ConversionPlan {
                    input: OperandRequirement::new(value.tensor_type.format.clone()),
                    output: OperandRequirement::new(output.tensor_type.format.clone()),
                    strategy: ConversionStrategy::DirectRetile,
                }),
                ..cast.clone()
            };
            values.push(value);
            if !super::rewrite::fusion_pays(
                "source FP8 output",
                producer.source,
                std::iter::once(previous)
                    .chain(identity_copies.iter().copied())
                    .chain([index])
                    .map(|i| &operations[i]),
                [&fused, &copy],
                values,
            ) {
                values.pop();
                continue;
            }
            operations[previous] = fused;
            operations[index] = copy;
            removed.extend(identity_copies);
            continue;
        }
        let mut replacement = producer.clone();
        replacement.results = cast.results.clone();
        if redistributed {
            replacement.inputs = cast.inputs.clone();
        }
        replacement.kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: kernel.clone(),
            operands: operands.clone(),
            product: None,
            output_aliases: Vec::new(),
        });
        let mut new_values = values.clone();
        let mut copies = vec![];
        if redistributed {
            // Row reductions require complete rows; pointwise families may
            // move across column shards. Place every row parameter on the
            // selected owners using the same broadcast rules as normal lowering.
            let rank = input.tensor_type.shape.0.len();
            let width = *input.tensor_type.shape.0.last().unwrap();
            if capability.complete_rows
                && input
                    .tensor_type
                    .format
                    .layout
                    .tiling
                    .axes
                    .iter()
                    .any(|axis| {
                        axis.axis.resolve(rank).is_err()
                            || (axis.axis.resolve(rank) == Ok(rank - 1)
                                && (axis.partitions != 1
                                    || !width.is_multiple_of(axis.padding_multiple)
                                    || !width.is_multiple_of(axis.shard_padding_multiple)))
                    })
            {
                continue;
            }
            let mut legal = true;
            for &parameter in &producer.inputs[1..] {
                let old = &values[parameter.index() as usize];
                let Some(tiling) =
                    implementation::pointwise_input_tiling(&old.tensor_type, &input.tensor_type)
                else {
                    legal = false;
                    break;
                };
                let mut value = old.clone();
                value.tensor_type.format.layout.tiling = tiling;
                value.tile_offset = input.tile_offset;
                if same_storage(old, &value) {
                    replacement.inputs.push(parameter);
                    continue;
                }
                value.id = MidValueId(new_values.len() as u32);
                value.storage_group = value.id;
                copies.push(MidOperation {
                    source: producer.source,
                    inputs: vec![parameter],
                    results: vec![value.id],
                    kind: MidOperationKind::Convert(ConversionPlan {
                        input: OperandRequirement::new(old.tensor_type.format.clone()),
                        output: OperandRequirement::new(value.tensor_type.format.clone()),
                        strategy: ConversionStrategy::DirectRetile,
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                });
                replacement.inputs.push(value.id);
                new_values.push(value);
            }
            if !legal {
                continue;
            }
        }
        if !super::rewrite::fusion_pays(
            "consumer FP8 output",
            producer.source,
            [&producer, &cast],
            std::iter::once(&replacement).chain(&copies),
            &new_values,
        ) {
            continue;
        }
        *values = new_values;
        if !copies.is_empty() {
            preparation.insert(index, copies);
        }
        if redistributed {
            let first = *identity_copies.last().expect("redistribution has a copy");
            operations[first].inputs[0] = producer.inputs[0];
        } else {
            removed.extend(identity_copies);
        }
        operations[index] = replacement;
        removed.insert(previous);
    }
    let changed = !removed.is_empty();
    apply_edits(operations, &removed, preparation);
    changed
}
