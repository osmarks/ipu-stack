//! Storage donation for shrinking casts with a bank-separated output prefix.
use crate::kernel::TileKernelSpec;
use crate::kernel::cast::{CAST_PREFIX_BYTES, CastChunks};
use crate::mid::{
    Compute, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId, OperandIndexing,
};
use crate::tensor::Precision;
use std::collections::BTreeSet;

impl MidProgram {
    pub(crate) fn reuse_cast_inputs(&mut self) {
        donate(&mut self.operations, &mut self.values, &self.outputs, false);
    }
}

fn donate(
    operations: &mut [MidOperation],
    values: &mut [MidValue],
    required: &[MidValueId],
    bound_outputs: bool,
) {
    // Repeat binds yielded storage to the previous iteration's input. A
    // displaced donor could then overlap that input during its producer.
    // Keep these bindings ordinary; internal casts can still donate.
    let mut bound = if bound_outputs {
        required.iter().copied().collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    for op in operations.iter().rev() {
        if !op.results.iter().any(|v| bound.contains(v)) {
            continue;
        }
        match &op.kind {
            MidOperationKind::Copy {
                reuse_local: true, ..
            } => bound.extend(op.inputs.iter().copied()),
            MidOperationKind::Compute(compute) => {
                let output_aliases = compute.output_aliases();
                for &(output, input) in output_aliases {
                    if bound.contains(&op.results[output]) {
                        bound.insert(op.inputs[input]);
                    }
                }
            }
            _ => {}
        }
    }
    let producers = super::rewrite::single_use_producers(operations, required);
    for index in 0..operations.len() {
        if let MidOperationKind::Repeat(repeat) = &mut operations[index].kind {
            donate(
                &mut repeat.body.operations,
                values,
                &repeat.body.yields,
                true,
            );
            continue;
        }
        let Some((input, output)) = super::rewrite::fp8_cast(&operations[index], values) else {
            continue;
        };
        if bound.contains(&output) {
            continue;
        }
        let Some(&producer) = producers.get(&input).filter(|&&p| p < index) else {
            continue;
        };
        // A fresh result inside this region is required: no parameters, Repeat
        // arguments, preexisting writable aliases, or escaped storage groups.
        let source = &values[input.index() as usize];
        let target = &values[output.index() as usize];
        tracing::debug!(?input, ?output, input_layout = ?source.tensor_type.format.layout,
            output_layout = ?target.tensor_type.format.layout, "considering cast storage donation");
        if source.owners != target.owners
            || source.tensor_type.shape != target.tensor_type.shape
            || source.tensor_type.format.layout != target.tensor_type.format.layout
            || values
                .iter()
                .filter(|v| v.storage_group == source.storage_group)
                .count()
                != 1
        {
            continue;
        }
        let fresh = match &operations[producer].kind {
            MidOperationKind::Copy { .. } => true,
            MidOperationKind::Compute(Compute::Sum { .. }) => false,
            MidOperationKind::Compute(compute) => {
                let output_aliases = compute.output_aliases();
                output_aliases.is_empty()
            }
            _ => false,
        };
        if !fresh {
            continue;
        }
        let Ok(shards) = source
            .tensor_type
            .format
            .layout
            .shard_extents(&source.tensor_type.shape)
        else {
            continue;
        };
        if shards.is_empty()
            || shards.iter().any(|(_, extents)| {
                let dimensions = extents
                    .iter()
                    .map(|e| e.physical_end - e.start)
                    .collect::<Vec<_>>();
                // Donation adds a prefix to the F16 allocation and removes a
                // separate FP8 result. Require a net saving on every shard.
                let output_bytes = dimensions
                    .iter()
                    .fold(1u64, |n, &d| n.saturating_mul(u64::from(d)));
                output_bytes <= u64::from(CAST_PREFIX_BYTES)
                    || CastChunks::new(source.tensor_type.format.layout.order, &dimensions)
                        .is_none()
            })
        {
            continue;
        }
        // A coordinate copy may otherwise alias an external parameter or an
        // earlier activation in low. Donation requires its own materialization.
        if let MidOperationKind::Copy { reuse_local, .. } = &mut operations[producer].kind {
            *reuse_local = false;
        }
        operations[index].kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Cast {
                from: Precision::F16,
                to: target.tensor_type.format.precision,
            },
            operands: vec![OperandIndexing::Elementwise { result: 0 }],
            output_aliases: vec![(0, 0)],
        });
        tracing::debug!(?input, ?output, "donated cast input storage");
        values[output.index() as usize].storage_group =
            values[input.index() as usize].storage_group;
    }
}

#[cfg(test)]
mod tests {
    use crate::estimate::MemoryPeaks;
    use crate::graph::{GraphInputKind, ValueId};
    use crate::low::CopyPolicy;
    use crate::mid::{CoordinateMapping, MidInput, MidRegion, MidRepeat};
    use crate::tensor::{AmpOrder, ElementOrder, Layout, TensorTiling, TensorType};
    use std::collections::BTreeMap;

    use super::*;

    fn fixture(order: ElementOrder, shape: &[u32]) -> MidProgram {
        let mut layout = Layout::row_major(TensorTiling::replicated(1));
        layout.order = order;
        let values = (0..3)
            .map(|index| MidValue {
                id: MidValueId(index),
                owners: crate::tensor::OwnerMap::default(),
                origin: ValueId::from_index(0),
                storage_group: MidValueId(index),
                tensor_type: TensorType::new(
                    shape.to_vec(),
                    if index == 2 {
                        Precision::F8F143 { scale_exponent: 0 }
                    } else {
                        Precision::F16
                    },
                    layout.clone(),
                ),
            })
            .collect();
        MidProgram {
            tile_count: 1,
            values,
            inputs: vec![MidInput {
                name: "parameter".into(),
                kind: GraphInputKind::Parameter,
                value: MidValueId(0),
            }],
            outputs: vec![MidValueId(2)],
            operations: vec![
                MidOperation {
                    site: None,
                    source: None,
                    inputs: vec![MidValueId(0)],
                    results: vec![MidValueId(1)],
                    kind: MidOperationKind::Copy {
                        policy: crate::CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Automatic,
                        mapping: CoordinateMapping::default(),
                        reuse_local: true,
                    },
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                },
                MidOperation {
                    site: None,
                    source: None,
                    inputs: vec![MidValueId(1)],
                    results: vec![MidValueId(2)],
                    kind: MidOperationKind::Compute(Compute::Kernel {
                        kernel: TileKernelSpec::Cast {
                            from: Precision::F16,
                            to: Precision::F8F143 { scale_exponent: 0 },
                        },
                        operands: vec![OperandIndexing::Elementwise { result: 0 }],
                        output_aliases: vec![],
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                },
            ],
            ..MidProgram::default()
        }
    }

    #[test]
    fn donation_preserves_parameters_and_cast_calls_never_overlap_unread_input() {
        for (order, shape) in [
            (ElementOrder::Amp(AmpOrder::Left), vec![164, 384]),
            (ElementOrder::RowMajor, vec![164, 384]),
            (ElementOrder::RowMajor, vec![65536]),
        ] {
            let mut mid = fixture(order, &shape);
            let before_memory = crate::estimate::analyze_mid(&mid, &BTreeMap::new())
                .unwrap()
                .1;
            let before = crate::estimate::operation_cost(&mid.operations[1], &mid.values).unwrap();
            mid.reuse_cast_inputs();
            let after = crate::estimate::operation_cost(&mid.operations[1], &mid.values).unwrap();
            assert!(after.0.total > before.0.total);
            assert_eq!(after.1.total(), 0);
            let after_memory = crate::estimate::analyze_mid(&mid, &BTreeMap::new())
                .unwrap()
                .1;
            assert_eq!(
                before_memory.total - after_memory.total,
                u64::from(shape.iter().product::<u32>() - CAST_PREFIX_BYTES)
            );
            let graph = crate::low::expand::expand_tiles(&mid, false).unwrap();
            let low = crate::low::lower_to_tiles(&graph, false);
            let placement = crate::place::place(&low).unwrap();
            let parameter = low.value_shards(low.inputs[0].value)[0];
            let output = low.value_shards(low.outputs[0])[0];
            let crate::ShardDefinition::ShiftedAlias {
                source: input,
                offset: -32768,
            } = low.shards[output.index() as usize].definition
            else {
                panic!("cast must donate its input")
            };
            assert_ne!(crate::storage_root(&low.shards, input), parameter);
            assert_eq!(
                placement.shard_addresses[&input],
                placement.shard_addresses[&output] + 32768
            );
            let mut written = 0;
            for run in &low.kernel_runs {
                run.call().unwrap();
                let src = &run.inputs[0];
                let dst = &run.outputs[0];
                let spans = |v: &crate::ShardView| {
                    let spans =
                        crate::view_byte_spans(&low.shards[v.shard.index() as usize], v).unwrap();
                    let base = placement.shard_addresses[&v.shard];
                    (
                        base + spans[0].offset,
                        base + spans.last().unwrap().offset + spans.last().unwrap().bytes,
                    )
                };
                let (a, b) = spans(src);
                let (c, d) = spans(dst);
                assert!(
                    b <= c || d <= a,
                    "each worker call must have disjoint buffers"
                );
                assert!(
                    (b - 1) / 32768 < c / 32768 || (d - 1) / 32768 < a / 32768,
                    "every call must use the simultaneous load/store fast path"
                );
                written += d - c;
            }
            assert_eq!(written, shape.iter().product::<u32>());
        }
    }

    #[test]
    fn donation_rejects_live_inputs_and_region_arguments() {
        let mut mid = fixture(ElementOrder::Amp(AmpOrder::Left), &[164, 384]);
        mid.outputs.push(MidValueId(1));
        let old = mid.clone();
        mid.reuse_cast_inputs();
        assert_eq!(mid, old);
        mid.outputs.pop();
        mid.operations.remove(0);
        let old = mid.clone();
        mid.reuse_cast_inputs();
        assert_eq!(mid, old);
    }

    #[test]
    fn physically_valid_donation_is_skipped_when_it_would_increase_storage() {
        let dimensions = [16384];
        assert!(CastChunks::new(ElementOrder::RowMajor, &dimensions).is_some());
        let mut mid = fixture(ElementOrder::RowMajor, &dimensions);
        let before = mid.clone();
        mid.reuse_cast_inputs();
        assert_eq!(mid, before);
    }

    #[test]
    fn repeat_keeps_internal_donation_but_protects_carried_output_storage() {
        let mut mid = fixture(ElementOrder::RowMajor, &[65536]);
        let mut argument = mid.values[0].clone();
        argument.id = MidValueId(3);
        argument.storage_group = argument.id;
        mid.values.push(argument);
        mid.operations[0].inputs[0] = MidValueId(3);
        let body = MidRegion {
            arguments: vec![MidValueId(3)],
            operations: std::mem::take(&mut mid.operations),
            yields: vec![],
            estimated_cycles: 0,
            peak_memory: MemoryPeaks::default(),
        };
        mid.outputs.clear();
        mid.operations.push(MidOperation {
            site: None,
            source: None,
            inputs: vec![MidValueId(0)],
            results: vec![],
            kind: MidOperationKind::Repeat(MidRepeat {
                count: 3,
                carried_inputs: 0,
                invariant_inputs: 1,
                iterated_inputs: vec![],
                body,
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        });
        mid.reuse_cast_inputs();
        let graph = crate::low::expand::expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        crate::place::place(&low).unwrap();
        assert!(
            low.shards
                .iter()
                .any(|s| matches!(s.definition, crate::ShardDefinition::ShiftedAlias { .. }))
        );
        assert_eq!(low.repeat_runs[0].count, 3);

        let mut mid = fixture(ElementOrder::RowMajor, &[65536]);
        let old = mid.clone();
        donate(&mut mid.operations, &mut mid.values, &mid.outputs, true);
        assert_eq!(mid, old);
        // An alias of a yield has the same restriction.
        let mut alias = mid.values[2].clone();
        alias.id = MidValueId(3);
        alias.storage_group = alias.id;
        mid.values.push(alias);
        mid.operations.push(MidOperation {
            site: None,
            source: None,
            inputs: vec![MidValueId(2)],
            results: vec![MidValueId(3)],
            kind: MidOperationKind::Copy {
                policy: crate::CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
                mapping: CoordinateMapping::default(),
                reuse_local: true,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        });
        let old = mid.clone();
        donate(&mut mid.operations, &mut mid.values, &[MidValueId(3)], true);
        assert_eq!(mid, old);
    }

    #[test]
    fn local_rearrangements_and_multiple_shards_use_the_corresponding_donor() {
        let mut mid = fixture(ElementOrder::Amp(AmpOrder::Left), &[2, 49152]);
        for value in &mut mid.values {
            value.tensor_type.format.layout.tiling = TensorTiling::linear(1, 32);
        }
        // Exercise a real local packing kernel before donation. An AMP-left
        // to AMP-left "rearrange" has no kernel and used to escape this test.
        mid.values[0].tensor_type.format.layout.order = ElementOrder::RowMajor;
        mid.operations[0].kind = MidOperationKind::Copy {
            mapping: CoordinateMapping::default(),
            reuse_local: false,
            policy: CopyPolicy::LocalKernel,
            packing: crate::PackingPolicy::Automatic,
        };
        mid.reuse_cast_inputs();
        let graph = crate::low::expand::expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        assert_eq!(low.value_shards(low.outputs[0]).len(), 2);
        let placement = crate::place::place(&low).unwrap();
        let casts = low
            .kernel_runs
            .iter()
            .filter(|run| matches!(run.kernel, TileKernelSpec::Cast { .. }))
            .collect::<Vec<_>>();
        assert!(casts.len() >= 2);
        for run in casts {
            let crate::ShardDefinition::ShiftedAlias {
                source: donor,
                offset: -32768,
            } = low.shards[run.outputs[0].shard.index() as usize].definition
            else {
                panic!("donated output")
            };
            let src = run.inputs[0].shard;
            if low.shards[src.index() as usize].definition != crate::ShardDefinition::Staging {
                assert_eq!(src, donor);
            }
            assert_eq!(
                placement.shard_addresses[&donor],
                placement.shard_addresses[&run.outputs[0].shard] + 32768
            );
        }
    }
}
