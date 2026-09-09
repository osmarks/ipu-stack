//! Fuse compatible whole-device primitives before physical expansion.
use super::*;

impl MidProgram {
    pub(super) fn with_elementwise_fusions(&self) -> Option<Self> {
        let mut result = self.clone();
        let residual =
            super::residual::fuse(&mut result.operations, &mut result.values, &result.outputs);
        if !fuse_region(&mut result.operations, &mut result.values, &result.outputs) && !residual {
            return None;
        }
        let (before, _) = crate::estimate::analyze_mid(self, &BTreeMap::new())?;
        let (after, peak) = crate::estimate::analyze_mid(&result, &BTreeMap::new())?;
        if after.total >= before.total {
            return None;
        }
        result.estimated_cycles = after.total;
        result.estimated_exchange_cycles = after.exchange;
        result.peak_memory = peak;
        Some(result)
    }
}

fn fuse_region(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = fuse_fp8_outputs(operations, values, required);
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= fuse_region(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    let mut users = BTreeMap::<MidValueId, BTreeSet<usize>>::new();
    let mut producers = BTreeMap::new();
    for (index, operation) in operations.iter().enumerate() {
        for &input in operation.read_values() {
            users.entry(input).or_default().insert(index);
        }
        for &output in &operation.results {
            producers.insert(output, index);
        }
    }
    for &output in required {
        users.entry(output).or_default().insert(usize::MAX);
    }
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let current = &operations[index];
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel,
            operands,
            product: None,
            ..
        }) = &current.kind
        else {
            continue;
        };
        let fused = match kernel {
            TileKernelSpec::Gelu => TileKernelSpec::BiasGelu,
            TileKernelSpec::LayerNorm => TileKernelSpec::AddLayerNorm,
            _ => continue,
        };
        if operands.iter().any(|window| !window.0.is_empty()) {
            continue;
        }
        let Some(&input) = current.inputs.first() else {
            continue;
        };
        let Some(&previous) = producers.get(&input) else {
            continue;
        };
        if previous >= index || users.get(&input).is_none_or(|uses| uses.len() != 1) {
            continue;
        }
        let add = &operations[previous];
        let MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::Add,
            operands: add_operands,
            product: None,
            ..
        }) = &add.kind
        else {
            continue;
        };
        if add_operands.len() != 2 || add_operands.iter().any(|window| !window.0.is_empty()) {
            continue;
        }
        let [output] = current.results.as_slice() else {
            continue;
        };
        let output_value = &values[output.index() as usize];
        let output_type = &output_value.tensor_type;
        let left = &values[add.inputs[0].index() as usize];
        let right = &values[add.inputs[1].index() as usize];
        if &left.tensor_type != output_type
            || left.tile_offset != output_value.tile_offset
            || output_type.format.precision != Precision::F16
            || output_type.format.layout.order != ElementOrder::RowMajor
            || values[input.index() as usize].tensor_type != *output_type
            || values[input.index() as usize].tile_offset != output_value.tile_offset
        {
            continue;
        }
        if !compatible_fusion(&fused, &left.tensor_type, &right.tensor_type, output_type)
            || (fused == TileKernelSpec::AddLayerNorm
                && right.tile_offset != output_value.tile_offset)
        {
            continue;
        }
        // Parameter copies may intervene. Delay the add only if none can
        // overwrite either source; other computation retains its ordering.
        let groups = [
            left.storage_group,
            right.storage_group,
            values[input.index() as usize].storage_group,
        ];
        if operations[previous + 1..index].iter().any(|step| {
            !matches!(
                step.kind,
                MidOperationKind::Primitive(Primitive::Copy { .. })
            ) || step
                .results
                .iter()
                .any(|id| groups.contains(&values[id.index() as usize].storage_group))
        }) {
            continue;
        }
        let mut inputs = add.inputs[..2].to_vec();
        inputs.extend_from_slice(&current.inputs[1..operands.len()]);
        let reuse_input = inputs.iter().position(|id| {
            let value = &values[id.index() as usize];
            value.storage_group == output_value.storage_group && value.tensor_type == *output_type
        });
        let arity = inputs.len();
        let mut replacement = current.clone();
        replacement.inputs = inputs;
        replacement.kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: fused,
            operands: vec![OperandWindow::default(); arity],
            product: None,
            output_aliases: reuse_input.map(|input| (0, input)).into_iter().collect(),
        });
        let prices = crate::estimate::operation_cost(add, values)
            .zip(crate::estimate::operation_cost(current, values))
            .zip(crate::estimate::operation_cost(&replacement, values));
        if prices.is_none_or(|(((a, _, _), (b, _, _)), (fused, _, _))| {
            fused.total >= a.total.saturating_add(b.total)
        }) {
            continue;
        }
        operations[index] = replacement;
        removed.insert(previous);
        changed = true;
    }
    let mut index = 0;
    operations.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    changed
}

pub(super) fn same_storage(a: &MidValue, b: &MidValue) -> bool {
    a.tile_offset == b.tile_offset
        && a.tensor_type.format.precision == b.tensor_type.format.precision
        && a.tensor_type.format.layout.order == b.tensor_type.format.layout.order
        && a.tensor_type.format.layout.memory_class == b.tensor_type.format.layout.memory_class
        && implementation::same_distribution(&a.tensor_type, &b.tensor_type)
}

/// Walk coordinate-preserving copies, optionally requiring identical storage.
/// Callers check extra readers and intervening writes before rewriting them.
pub(super) fn producer_through_copies(
    mut value: MidValueId,
    operations: &[MidOperation],
    values: &[MidValue],
    identity_only: bool,
) -> Option<(MidValueId, usize, Vec<usize>)> {
    let mut copies = vec![];
    loop {
        let index = operations.iter().rposition(|op| op.results == [value])?;
        let op = &operations[index];
        let identity = match &op.kind {
            MidOperationKind::Convert(_) => true,
            MidOperationKind::Primitive(Primitive::Copy { mapping, .. }) => {
                *mapping == CoordinateMapping::default()
            }
            _ => false,
        };
        if !identity
            || op.inputs.len() != 1
            || values[value.index() as usize].tensor_type.shape
                != values[op.inputs[0].index() as usize].tensor_type.shape
            || values[value.index() as usize].tensor_type.format.precision
                != values[op.inputs[0].index() as usize]
                    .tensor_type
                    .format
                    .precision
            || (identity_only
                && !same_storage(
                    &values[value.index() as usize],
                    &values[op.inputs[0].index() as usize],
                ))
        {
            return Some((value, index, copies));
        }
        copies.push(index);
        value = op.inputs[0];
    }
}

// A producer may write a cast's explicit result when its F16 intermediate
// has no other readers. Keep the separate path whenever it costs less.
fn fuse_fp8_outputs(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut removed = BTreeSet::new();
    let mut preparation = BTreeMap::<usize, Vec<MidOperation>>::new();
    for index in 0..operations.len() {
        let cast = operations[index].clone();
        let local_cast = match &cast.kind {
            MidOperationKind::Convert(plan) => plan.strategy == ConversionStrategy::LocalKernel,
            MidOperationKind::Primitive(Primitive::Compute {
                kernel: TileKernelSpec::Cast { .. },
                operands,
                product: None,
                output_aliases,
            }) => operands.len() == 1 && operands[0].0.is_empty() && output_aliases.is_empty(),
            _ => false,
        };
        if !local_cast
            || cast.inputs.len() != 1
            || cast.results.len() != 1
            || values[cast.inputs[0].index() as usize]
                .tensor_type
                .format
                .precision
                != Precision::F16
            || !matches!(
                values[cast.results[0].index() as usize]
                    .tensor_type
                    .format
                    .precision,
                Precision::F8F143 { .. }
            )
        {
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
        if !matches!(kernel, TileKernelSpec::Gelu | TileKernelSpec::LayerNorm)
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
        if input.tile_offset != output.tile_offset
            || input.tensor_type.format.layout.order != ElementOrder::RowMajor
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
        if expected.is_none_or(|layout| {
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
                    .map(|a| a.extents_are_multiple_of(4))
            })
            != Some(true)
        {
            continue;
        }
        // Parameter copies may intervene, provided they cannot overwrite inputs.
        let groups = producer
            .inputs
            .iter()
            .map(|v| values[v.index() as usize].storage_group)
            .collect::<BTreeSet<_>>();
        if operations[previous + 1..index]
            .iter()
            .enumerate()
            .any(|(i, op)| {
                !identity_copies.contains(&(previous + 1 + i))
                    && (!matches!(
                        op.kind,
                        MidOperationKind::Primitive(Primitive::Copy { .. })
                            | MidOperationKind::Convert(_)
                    ) || op
                        .results
                        .iter()
                        .any(|v| groups.contains(&values[v.index() as usize].storage_group)))
            })
        {
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
        if redistributed && *kernel == TileKernelSpec::LayerNorm {
            // LN commutes only with redistribution of complete rows. Use the
            // normal broadcast tiling to place affine parameters on those owners.
            let rank = input.tensor_type.shape.0.len();
            let width = *input.tensor_type.shape.0.last().unwrap();
            if producer.inputs.len() != 3
                || input
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
                        input: OperandRequirement::new(old.tensor_type.format.clone(), 8),
                        output: OperandRequirement::new(value.tensor_type.format.clone(), 8),
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
        let copy_cost = copies.iter().try_fold(0u64, |total, op| {
            crate::estimate::operation_cost(op, &new_values)
                .map(|(cost, _, _)| total.saturating_add(cost.total))
        });
        let prices = crate::estimate::operation_cost(&producer, values)
            .zip(crate::estimate::operation_cost(&cast, values))
            .zip(crate::estimate::operation_cost(&replacement, &new_values));
        if prices.is_none_or(|(((a, _, _), (b, _, _)), (c, _, _))| {
            copy_cost.is_none_or(|extra| {
                c.total.saturating_add(extra) >= a.total.saturating_add(b.total)
            })
        }) {
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
    *operations = std::mem::take(operations)
        .into_iter()
        .enumerate()
        .flat_map(|(index, op)| {
            let mut prefix = preparation.remove(&index).unwrap_or_default();
            if !removed.contains(&index) {
                prefix.push(op);
            }
            prefix
        })
        .collect();
    changed
}

// Shared with the optimistic regional search: known fusion is not a missing kernel.
pub(super) fn compatible_fusion(
    kernel: &TileKernelSpec,
    left: &TensorType,
    right: &TensorType,
    output: &TensorType,
) -> bool {
    if left != output
        || output.format.precision != Precision::F16
        || right.format.precision != Precision::F16
        || output.format.layout.order != ElementOrder::RowMajor
    {
        return false;
    }
    match kernel {
        TileKernelSpec::BiasGelu => {
            right.shape.0.last() == output.shape.0.last()
                && !right.shape.0.is_empty()
                && right.shape.0[..right.shape.0.len() - 1]
                    .iter()
                    .all(|&size| size == 1)
        }
        TileKernelSpec::AddLayerNorm => right == output,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gelu_moves_through_retile_into_fp8_consumer_owners() {
        let source = TensorType::new([12, 1152], Precision::F16, Layout::row_sharded(1));
        let retiled = TensorType::new([12, 1152], Precision::F16, Layout::row_sharded(12));
        let mut packed = retiled.clone();
        packed.format.precision = Precision::F8F143 { scale_exponent: -4 };
        packed.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        let values = [
            source.clone(),
            source.clone(),
            retiled.clone(),
            packed.clone(),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, tensor_type)| {
            let id = MidValueId(index as u32);
            MidValue {
                id,
                tile_offset: 0,
                tensor_type,
                origin: ValueId::from_index(0),
                storage_group: id,
            }
        })
        .collect();
        let operation = |input, result, kind| MidOperation {
            source: None,
            inputs: vec![MidValueId(input)],
            results: vec![MidValueId(result)],
            kind,
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let mut program = MidProgram {
            tile_count: 12,
            values,
            operations: vec![
                operation(
                    0,
                    1,
                    MidOperationKind::Primitive(Primitive::Compute {
                        kernel: TileKernelSpec::Gelu,
                        operands: vec![OperandWindow::default()],
                        product: None,
                        output_aliases: vec![],
                    }),
                ),
                operation(
                    1,
                    2,
                    MidOperationKind::Convert(ConversionPlan {
                        input: OperandRequirement::new(source.format, 8),
                        output: OperandRequirement::new(retiled.format.clone(), 8),
                        strategy: ConversionStrategy::DirectRetile,
                    }),
                ),
                operation(
                    2,
                    3,
                    MidOperationKind::Convert(ConversionPlan {
                        input: OperandRequirement::new(retiled.format, 8),
                        output: OperandRequirement::new(packed.format, 8),
                        strategy: ConversionStrategy::LocalKernel,
                    }),
                ),
            ],
            outputs: vec![MidValueId(3)],
            ..MidProgram::default()
        };
        let fused = program.with_elementwise_fusions().unwrap();
        assert_eq!(fused.operations.len(), 2);
        assert_eq!(fused.operations[0].inputs, [MidValueId(0)]);
        assert_eq!(fused.operations[1].inputs, [MidValueId(2)]);
        assert!(matches!(
            fused.operations[1].kind,
            MidOperationKind::Primitive(Primitive::Compute {
                kernel: TileKernelSpec::Gelu,
                ..
            })
        ));
        let graph = crate::expand_tiles(&fused).unwrap();
        for run in &graph.kernel_runs {
            crate::validate_kernel_run(run).unwrap();
        }
        let mut norm = program.clone();
        for _ in 0..2 {
            let id = MidValueId(norm.values.len() as u32);
            norm.values.push(MidValue {
                id,
                storage_group: id,
                tile_offset: 0,
                origin: ValueId::from_index(0),
                tensor_type: TensorType::new(
                    [1152],
                    Precision::F16,
                    Layout::row_major(TensorTiling::replicated(1)),
                ),
            });
            norm.operations[0].inputs.push(id);
        }
        norm.operations[0].kind = MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::LayerNorm,
            operands: vec![OperandWindow::default(); 3],
            product: None,
            output_aliases: vec![],
        });
        let fused_norm = norm.with_elementwise_fusions().unwrap();
        assert_eq!(fused_norm.operations.len(), 4); // activation, gamma, beta copies + LN
        let last = fused_norm.operations.last().unwrap();
        assert_eq!(last.inputs.len(), 3);
        assert!(matches!(
            last.kind,
            MidOperationKind::Primitive(Primitive::Compute {
                kernel: TileKernelSpec::LayerNorm,
                ..
            })
        ));
        let graph = crate::expand_tiles(&fused_norm).unwrap();
        for run in &graph.kernel_runs {
            crate::validate_kernel_run(run).unwrap();
        }
        program.outputs.push(MidValueId(1));
        assert!(program.with_elementwise_fusions().is_none());
    }

    #[test]
    fn fp8_output_fusion_preserves_other_users_and_prices_packed_rows() {
        for norm in [false, true] {
            for rows in [1, 3] {
                let source = TensorType::new([rows, 1152], Precision::F16, Layout::row_sharded(1));
                let mut output = source.clone();
                output.format.precision = Precision::F8F143 { scale_exponent: -4 };
                output.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
                let mut values = vec![];
                for tensor in [
                    source.clone(),
                    source.clone(),
                    source.clone(),
                    source.clone(),
                    output.clone(),
                ] {
                    let id = MidValueId(values.len() as u32);
                    values.push(MidValue {
                        id,
                        tile_offset: 0,
                        tensor_type: tensor,
                        origin: ValueId::from_index(0),
                        storage_group: id,
                    });
                }
                let inputs = if norm {
                    vec![MidValueId(0), MidValueId(1), MidValueId(2)]
                } else {
                    vec![MidValueId(0)]
                };
                let producer = MidOperation {
                    source: None,
                    results: vec![MidValueId(3)],
                    kind: MidOperationKind::Primitive(Primitive::Compute {
                        kernel: if norm {
                            TileKernelSpec::LayerNorm
                        } else {
                            TileKernelSpec::Gelu
                        },
                        operands: vec![OperandWindow::default(); inputs.len()],
                        product: None,
                        output_aliases: Vec::new(),
                    }),
                    inputs,
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                let mut identity = values[3].clone();
                identity.id = MidValueId(5);
                identity
                    .tensor_type
                    .format
                    .layout
                    .tiling
                    .axes
                    .push(AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        1,
                        4,
                        Padding::Reject,
                    ));
                let identity_format = identity.tensor_type.format.clone();
                values.push(identity);
                let copy = MidOperation {
                    source: None,
                    inputs: vec![MidValueId(3)],
                    results: vec![MidValueId(5)],
                    kind: MidOperationKind::Convert(ConversionPlan {
                        input: OperandRequirement::new(source.format.clone(), 8),
                        output: OperandRequirement::new(identity_format.clone(), 8),
                        strategy: ConversionStrategy::DirectRetile,
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                let mut cast = MidOperation {
                    source: None,
                    inputs: vec![MidValueId(5)],
                    results: vec![MidValueId(4)],
                    kind: MidOperationKind::Convert(ConversionPlan {
                        input: OperandRequirement::new(identity_format, 8),
                        output: OperandRequirement::new(output.format, 8),
                        strategy: ConversionStrategy::LocalKernel,
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                if rows > 1 {
                    cast.kind = MidOperationKind::Primitive(Primitive::Compute {
                        kernel: TileKernelSpec::Cast {
                            from: Precision::F16,
                            to: Precision::F8F143 { scale_exponent: -4 },
                        },
                        operands: vec![OperandWindow::default()],
                        product: None,
                        output_aliases: Vec::new(),
                    });
                }
                let mut program = MidProgram {
                    tile_count: 1,
                    values,
                    operations: vec![producer, copy, cast],
                    outputs: vec![MidValueId(4)],
                    ..MidProgram::default()
                };
                let fused = program.with_elementwise_fusions();
                assert_eq!(fused.is_some(), !norm || rows == 1);
                if let Some(fused) = fused {
                    let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
                    assert_eq!(low.kernel_runs.len(), 1);
                    crate::validate_kernel_run(&low.kernel_runs[0]).unwrap();
                    let build = crate::KernelBuildPlan::from_program(&low).unwrap();
                    let call = build.call(&low.kernel_runs[0]).unwrap();
                    let mut expected = vec![rows, 1152, (-4i32) as u32, 1];
                    if !norm {
                        expected.extend([1152, 1152]);
                    }
                    assert_eq!(call.arguments, expected);
                }
                program.outputs.push(MidValueId(3));
                assert!(program.with_elementwise_fusions().is_none());
            }
        }
    }

    #[test]
    fn fusion_preserves_live_add_results_and_matches_kernel_contracts() {
        // Short rows retain a launch-saving fused alternative. For wide
        // rows the new standalone add may be cheaper than recomputing it
        // inside each normalization pass.
        for norm in [false, true] {
            for keep_sum in [false, true] {
                let mut graph = ComputeGraph::new();
                let x = graph.host_input("x", [2, 4, 48]).unwrap();
                let rhs = graph
                    .host_input("rhs", if norm { vec![2, 4, 48] } else { vec![1, 1, 48] })
                    .unwrap();
                let sum = graph.add(x, rhs).unwrap();
                let mut config = PipelineConfig::new(4)
                    .with_automatic_input(x, Precision::F16)
                    .with_automatic_input(rhs, Precision::F16);
                let y = if norm {
                    let gamma = graph.parameter("gamma", [1, 1, 48]).unwrap();
                    let beta = graph.parameter("beta", [1, 1, 48]).unwrap();
                    config = config
                        .with_automatic_input(gamma, Precision::F16)
                        .with_automatic_input(beta, Precision::F16);
                    graph.layer_norm(sum, gamma, beta).unwrap()
                } else {
                    graph.gelu(sum).unwrap()
                };
                graph
                    .set_outputs(if keep_sum { vec![sum, y] } else { vec![y] })
                    .unwrap();
                let mid = implementation::resolve(lower(&graph, &config, &Ipu21CostModel).unwrap())
                    .unwrap();
                let fused = mid.with_elementwise_fusions();
                assert_eq!(fused.is_some(), !keep_sum, "norm={norm}");
                if let Some(fused) = fused {
                    assert!(
                        fused.estimated_cycles
                            < crate::estimate::analyze_mid(&mid, &BTreeMap::new())
                                .unwrap()
                                .0
                                .total
                    );
                    let low = crate::lower_to_tiles(
                        &crate::low::expand::expand_tiles(&fused, false).unwrap(),
                        false,
                    );
                    let mut count = 0;
                    for run in &low.kernel_runs {
                        if matches!(
                            run.kernel,
                            TileKernelSpec::BiasGelu | TileKernelSpec::AddLayerNorm
                        ) {
                            crate::validate_kernel_run(run).unwrap();
                            count += 1;
                        }
                    }
                    assert_eq!(count, 4);
                }
            }
        }
    }
}
