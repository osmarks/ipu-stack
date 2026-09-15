//! Donate dead FP16 scratch to FP8 outputs, then expand safe bank-separated calls.
//! This pass owns the storage decision; kernel::cast owns the chunk geometry.

use crate::kernel::cast::{CAST_PREFIX_BYTES, CastChunks};
use crate::low::storage::shard_storage_bytes;
use crate::low::*;
use crate::{MidOperationKind, Precision};
use std::collections::BTreeMap;

pub(super) fn donate(program: &mut TileGraph) -> ExpansionResult<()> {
    let mut uses = crate::low::uses::StorageUses::analyze(program);
    let roots = &uses.roots;
    let candidates = program
        .body
        .walk()
        .enumerate()
        .filter_map(|(index, operation)| {
            if let BlockOperation::Compute { run, .. } = operation
                && matches!(
                    program.kernel_runs[run.0 as usize].kernel,
                    MidOperationKind::Cast {
                        from: Precision::F16,
                        to: Precision::F8F143 { .. }
                    }
                )
            {
                Some((*run, index + 1))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut replacements = BTreeMap::new();
    let mut metadata = Vec::new();
    for (id, index) in candidates {
        let call = program.kernel_runs[id.0 as usize].clone();
        let ([input], [output]) = (call.inputs.as_slice(), call.outputs.as_slice()) else {
            continue;
        };
        let source = &program.shards[input.shard.index() as usize];
        let target = &program.shards[output.shard.index() as usize];
        let from = roots[input.shard.index() as usize];
        let to = roots[output.shard.index() as usize];
        // Read-only alias elimination may have exposed a shared or persistent
        // donor. Require independent, fully selected scratch on both sides.
        if from == to
            || uses.allocations[from].aliases != 1
            || uses.allocations[to].aliases != 1
            || uses.allocations[from].first == Some(0)
            || uses.allocations[from].last == usize::MAX
            || uses.allocations[from].boundary
            || uses.allocations[to].boundary
            || uses.allocations[from].writes == 0
            || uses.allocations[to].writes != 1
            || uses.allocations[from].last != index
            || input.extents != source.extents
            || output.extents != target.extents
            || input.extents != output.extents
            || source.tensor_type.format.layout.order != target.tensor_type.format.layout.order
            || source.tensor_type.format.layout.memory_class
                != target.tensor_type.format.layout.memory_class
            || shard_storage_bytes(target)? <= CAST_PREFIX_BYTES
        {
            continue;
        }
        let dimensions = input
            .extents
            .iter()
            .map(|e| e.physical_end - e.start)
            .collect::<Vec<_>>();
        let Some(chunks) = CastChunks::new(target.tensor_type.format.layout.order, &dimensions)
        else {
            continue;
        };
        program.shards[output.shard.index() as usize].definition = ShardDefinition::ShiftedAlias {
            source: input.shard,
            offset: -(CAST_PREFIX_BYTES as i32),
        };
        uses.allocations[from].aliases += 1;
        uses.allocations[to].aliases += 1;
        let mut calls = Vec::new();
        for (start, end) in chunks.ranges {
            let (mut input, mut output) = (input.clone(), output.clone());
            for view in [&mut input, &mut output] {
                let extent = &mut view.extents[chunks.axis];
                let base = extent.start;
                extent.start = base + start;
                extent.physical_end = base + end;
                extent.logical_end = extent
                    .logical_end
                    .min(extent.physical_end)
                    .max(extent.start);
            }
            calls.push(KernelRun::bind(
                call.provenance,
                call.kernel.clone(),
                vec![input],
                vec![output],
                &program.shards,
                &mut metadata,
            )?);
        }
        let mut calls = calls.into_iter();
        program.kernel_runs[id.0 as usize] =
            calls.next().ok_or(ExpansionError::InvalidOperatorPlan)?;
        let mut ids = vec![id];
        for call in calls {
            ids.push(KernelRunId(
                u32::try_from(program.kernel_runs.len()).map_err(|_| ExpansionError::IdOverflow)?,
            ));
            program.kernel_runs.push(call);
        }
        replacements.insert(id, ids);
    }
    let mut regions = vec![&mut program.body];
    while let Some(region) = regions.pop() {
        for operation in std::mem::take(&mut region.operations) {
            if let BlockOperation::Compute { tile, run } = operation
                && let Some(calls) = replacements.get(&run)
            {
                region.operations.extend(
                    calls
                        .iter()
                        .map(|&run| BlockOperation::Compute { tile, run }),
                );
            } else {
                region.operations.push(operation);
            }
        }
        for operation in &mut region.operations {
            if let BlockOperation::Repeat(repeat) = operation {
                regions.push(&mut repeat.body);
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use crate::graph::{GraphInputKind, ValueId};
    use crate::mid::{CoordinateMapping, MidInput, MidRegion, MidRepeat};
    use crate::CopyPolicy;
    use crate::tensor::{AmpOrder, ElementOrder, Layout, TensorTiling, TensorType};

    use super::*;
    use crate::{MidOperation, MidProgram, MidValue, MidValueId, OperandIndexing};

    fn fixture(order: ElementOrder, shape: &[u32]) -> MidProgram {
        let mut layout = Layout::row_major(TensorTiling::replicated(1));
        layout.order = order;
        let values = (0..3)
            .map(|index| MidValue {
                id: MidValueId::from_index(index),
                owners: crate::tensor::OwnerMap::default(),
                origin: ValueId::from_index(0),
                storage_group: MidValueId::from_index(index),
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
                value: MidValueId::from_index(0),
            }],
            outputs: vec![MidValueId::from_index(2)],
            operations: vec![
                MidOperation {
                    source: None,
                    inputs: vec![MidValueId::from_index(0)],
                    results: vec![MidValueId::from_index(1)],
                    kind: MidOperationKind::Gelu,
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: Vec::new(),
                    output_windows: Vec::new(),
                },
                MidOperation {
                    source: None,
                    inputs: vec![MidValueId::from_index(1)],
                    results: vec![MidValueId::from_index(2)],
                    kind: MidOperationKind::Cast {
                        from: Precision::F16,
                        to: Precision::F8F143 { scale_exponent: 0 },
                    },
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: vec![],
                    output_windows: Vec::new(),
                },
            ],
            ..MidProgram::default()
        }
    }

    fn expand(mid: &MidProgram, enabled: bool) -> std::sync::Arc<TileGraph> {
        crate::low::expand::expand_tiles_cached(mid, false, enabled, std::sync::Arc::default())
            .unwrap()
    }

    #[test]
    fn checkpoint_reads_extend_donor_lifetimes() {
        for after_cast in [false, true] {
            let mid = fixture(ElementOrder::RowMajor, &[65536]);
            let mut graph = (*expand(&mid, false)).clone();
            let checkpoint = serde_json::from_str("0").unwrap();
            graph.checkpoints = vec![(checkpoint, vec![MidValueId::from_index(1)])];
            let cast = graph.body.operations.iter().position(|op| matches!(op,
                BlockOperation::Compute { run, .. }
                    if matches!(graph.kernel_runs[run.0 as usize].kernel, MidOperationKind::Cast { .. })
            )).unwrap();
            graph.body.operations.insert(
                cast + usize::from(after_cast),
                BlockOperation::Checkpoint(checkpoint, 0),
            );
            donate(&mut graph).unwrap();
            assert_eq!(
                graph
                    .shards
                    .iter()
                    .any(|shard| matches!(shard.definition, ShardDefinition::ShiftedAlias { .. })),
                !after_cast
            );
            crate::place(&crate::low::lower_to_tiles(
                &std::sync::Arc::new(graph),
                false,
            ))
            .unwrap();
        }
    }

    #[test]
    fn repeat_reentry_keeps_outer_donors_live() {
        for producer_inside in [false, true] {
            let mid = fixture(ElementOrder::RowMajor, &[65536]);
            let mut graph = (*expand(&mid, false)).clone();
            let body = graph
                .body
                .operations
                .split_off(usize::from(!producer_inside));
            graph
                .body
                .operations
                .push(BlockOperation::Repeat(Box::new(BlockRepeat {
                    provenance: graph.kernel_runs[0].provenance,
                    count: 2,
                    // A direct capture still needs protection across iterations.
                    bindings: vec![],
                    body: BlockRegion { operations: body },
                })));
            donate(&mut graph).unwrap();
            assert_eq!(
                graph
                    .shards
                    .iter()
                    .any(|shard| matches!(shard.definition, ShardDefinition::ShiftedAlias { .. })),
                producer_inside
            );
            crate::place(&crate::low::lower_to_tiles(
                &std::sync::Arc::new(graph),
                false,
            ))
            .unwrap();
        }
    }

    #[test]
    fn nested_repeat_bindings_propagate_mutability() {
        let mid = fixture(ElementOrder::RowMajor, &[65536]);
        let mut graph = (*expand(&mid, false)).clone();
        let source = graph.value_views(MidValueId::from_index(0))[0].shard;
        let mut arguments = Vec::new();
        for _ in 0..2 {
            let mut shard = graph.shards[source.index() as usize].clone();
            shard.id = BlockValueId::from_index(graph.shards.len() as u32);
            shard.definition = ShardDefinition::Staging;
            arguments.push(shard.id);
            graph.shards.push(shard);
        }
        let copy = crate::kernel::CopyRun::bind(
            LocalCopy {
                source,
                destination: arguments[1],
                source_offset: 0,
                destination_offset: 0,
                bytes: 32,
                pattern: CopyPattern::Contiguous,
            },
            &graph.shards,
        )
        .unwrap();
        let id = LocalCopyId(graph.local_copies.len() as u32);
        graph.local_copies.push(copy);
        let mut body = BlockRegion {
            operations: vec![BlockOperation::Copy { tile: 0, copy: id }],
        };
        for (input, argument, iterated) in [
            (arguments[0], arguments[1], true),
            (source, arguments[0], false),
        ] {
            body = BlockRegion {
                operations: vec![BlockOperation::Repeat(Box::new(BlockRepeat {
                    provenance: graph.kernel_runs[0].provenance,
                    count: 2,
                    bindings: vec![BlockRepeatBinding {
                        tile: 0,
                        carried: vec![],
                        invariants: if iterated {
                            vec![]
                        } else {
                            vec![RepeatInvariant { input, argument }]
                        },
                        iterated: if iterated {
                            vec![RepeatIterated {
                                inputs: vec![input; 2],
                                argument,
                            }]
                        } else {
                            vec![]
                        },
                    }],
                    body,
                }))],
            };
        }
        graph.body = body;
        let uses = crate::low::uses::StorageUses::analyze(&graph);
        for id in [source, arguments[0], arguments[1]] {
            let allocation = &uses.allocations[uses.roots[id.index() as usize]];
            assert!(
                !allocation.read_only,
                "a nested argument write can modify {id:?}"
            );
            assert!(allocation.boundary);
        }
        assert_eq!(
            uses.allocations[uses.roots[source.index() as usize]].last,
            usize::MAX
        );
    }

    #[test]
    fn donation_uses_each_tiles_actual_lifetime() {
        let mut mid = fixture(ElementOrder::RowMajor, &[98304]);
        mid.tile_count = 2;
        for value in &mut mid.values {
            value.tensor_type.format.layout.tiling =
                TensorTiling::sharded(crate::TensorAxis::FromStart(0), 2);
        }
        let mut graph = (*expand(&mid, false)).clone();
        let input = graph.value_views(MidValueId::from_index(1))[0].clone();
        let mut extra = graph.shards[input.shard.index() as usize].clone();
        let tile = extra.tile;
        extra.id = BlockValueId::from_index(graph.shards.len() as u32);
        extra.definition = ShardDefinition::Staging;
        let output = ShardView {
            shard: extra.id,
            extents: extra.extents.clone(),
        };
        graph.shards.push(extra);
        let call = KernelRun::bind(
            graph.kernel_runs[0].provenance,
            MidOperationKind::Gelu,
            vec![input],
            vec![output],
            &graph.shards,
            &mut vec![],
        )
        .unwrap();
        let run = KernelRunId(graph.kernel_runs.len() as u32);
        graph.kernel_runs.push(call);
        graph
            .body
            .operations
            .push(BlockOperation::Compute { tile, run });
        donate(&mut graph).unwrap();
        for output in graph.value_views(MidValueId::from_index(2)) {
            let shard = &graph.shards[output.shard.index() as usize];
            assert_eq!(
                matches!(shard.definition, ShardDefinition::ShiftedAlias { .. }),
                shard.tile != tile
            );
        }
        crate::place::place(&crate::low::lower_to_tiles(
            &std::sync::Arc::new(graph),
            false,
        ))
        .unwrap();
    }

    #[test]
    fn rejects_live_shared_and_unprofitable_donors() {
        for variant in 0..4 {
            let mut mid = fixture(ElementOrder::RowMajor, &[65536]);
            match variant {
                0 => mid.outputs.push(MidValueId::from_index(1)),
                1 => {
                    mid.operations.remove(0);
                    mid.operations[0].inputs[0] = MidValueId::from_index(0);
                }
                2 => {
                    mid.operations[0].kind = MidOperationKind::Copy {
                        mapping: CoordinateMapping::default(),
                        policy: CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Automatic,
                    };
                    mid.operations[0].operands.clear();
                }
                _ => {
                    for value in &mut mid.values {
                        value.tensor_type.shape.0 = vec![16384];
                    }
                }
            }
            let graph = expand(&mid, true);
            assert!(
                !graph
                    .shards
                    .iter()
                    .any(|s| matches!(s.definition, ShardDefinition::ShiftedAlias { .. }))
            );
        }
        let mid = fixture(ElementOrder::RowMajor, &[65536]);
        let disabled = expand(&mid, false);
        assert!(
            !disabled
                .shards
                .iter()
                .any(|s| matches!(s.definition, ShardDefinition::ShiftedAlias { .. }))
        );
    }

    #[test]
    fn repeat_allows_internal_donation_but_protects_carried_results() {
        for carried in [false, true] {
            let mut mid = fixture(ElementOrder::RowMajor, &[65536]);
            let mut argument = mid.values[0].clone();
            argument.id = MidValueId::from_index(3);
            argument.storage_group = argument.id;
            mid.values.push(argument);
            mid.operations[0].inputs[0] = MidValueId::from_index(3);
            let mut body = MidRegion {
                arguments: vec![MidValueId::from_index(3)],
                operations: std::mem::take(&mut mid.operations),
                yields: vec![],
            };
            let mut inputs = vec![MidValueId::from_index(0)];
            let mut results = vec![];
            if carried {
                for index in 4..7 {
                    let mut value = mid.values[2].clone();
                    value.id = MidValueId::from_index(index);
                    value.storage_group = value.id;
                    mid.values.push(value);
                }
                mid.inputs.push(MidInput {
                    name: "initial".into(),
                    kind: GraphInputKind::Host,
                    value: MidValueId::from_index(4),
                });
                inputs.insert(0, MidValueId::from_index(4));
                body.arguments.insert(0, MidValueId::from_index(5));
                body.yields.push(MidValueId::from_index(2));
                results.push(MidValueId::from_index(6));
            }
            mid.outputs = results.clone();
            mid.operations.push(MidOperation {
                source: None,
                inputs,
                results,
                kind: MidOperationKind::Repeat(MidRepeat {
                    count: 3,
                    carried_inputs: usize::from(carried),
                    invariant_inputs: 1,
                    iterated_inputs: vec![],
                    body,
                }),
                operands: vec![],
                output_aliases: vec![],
                output_windows: vec![],
            });
            let graph = expand(&mid, true);
            assert_eq!(
                graph
                    .shards
                    .iter()
                    .any(|s| matches!(s.definition, ShardDefinition::ShiftedAlias { .. })),
                !carried
            );
            let low = crate::low::lower_to_tiles(&graph, false);
            crate::place::place(&low).unwrap();
            assert_eq!(low.repeat_runs[0].count, 3);
        }
    }
    #[test]
    fn donation_preserves_parameters_and_cast_calls_never_overlap_unread_input() {
        for (order, shape) in [
            (ElementOrder::Amp(AmpOrder::Left), vec![164, 384]),
            (ElementOrder::RowMajor, vec![164, 384]),
            (ElementOrder::RowMajor, vec![65536]),
        ] {
            let mid = fixture(order, &shape);
            let graph = expand(&mid, true);
            let low = crate::low::lower_to_tiles(&graph, false);
            let placement = crate::place::place(&low).unwrap();
            let parameter = low.value_views(low.inputs[0].value)[0].shard;
            let output = low.value_views(low.outputs[0])[0].shard;
            let crate::ShardDefinition::ShiftedAlias {
                source: input,
                offset: -32768,
            } = low.shards[output.index() as usize].definition
            else {
                panic!("cast must donate its input")
            };
            assert_ne!(
                crate::low::storage::storage_root(&low.shards, input),
                parameter
            );
            assert_eq!(
                placement.shard_addresses[&input],
                placement.shard_addresses[&output] + 32768
            );
            let mut metadata = Vec::new();
            let sample = low
                .kernel_runs
                .iter()
                .find(|r| matches!(r.kernel, MidOperationKind::Cast { .. }))
                .unwrap();
            for donated in [false, true, false] {
                let mut shards = low.shards.clone();
                if !donated {
                    shards[output.index() as usize].definition = crate::ShardDefinition::Staging;
                }
                let run = crate::KernelRun::bind(
                    sample.provenance,
                    sample.kernel.clone(),
                    sample.inputs.clone(),
                    sample.outputs.clone(),
                    &shards,
                    &mut metadata,
                )
                .unwrap();
                assert_eq!(
                    run.accesses(&shards)
                        .find(|(id, _)| *id == output)
                        .unwrap()
                        .1
                        .alignment,
                    if donated { CAST_PREFIX_BYTES } else { 8 }
                );
            }
            assert_eq!(
                metadata.len(),
                1,
                "storage changes must not duplicate format metadata"
            );
            let mut written = 0;
            for run in low
                .kernel_runs
                .iter()
                .filter(|r| matches!(r.kernel, MidOperationKind::Cast { .. }))
            {
                run.call(None).unwrap();
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
            policy: CopyPolicy::LocalKernel,
            packing: crate::PackingPolicy::Automatic,
        };
        let graph = expand(&mid, true);
        let low = crate::low::lower_to_tiles(&graph, false);
        assert_eq!(low.value_views(low.outputs[0]).len(), 2);
        let placement = crate::place::place(&low).unwrap();
        let casts = low
            .kernel_runs
            .iter()
            .filter(|run| matches!(run.kernel, MidOperationKind::Cast { .. }))
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
