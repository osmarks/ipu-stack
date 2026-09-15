use super::*;
use crate::mid::MidOperationKind;

use crate::{
    CoordinateMapping, GraphInputKind, MidInput, MidValue, OperandIndexing, TensorAxis, ValueId,
};

fn copied_columns(columns: u32) -> MidProgram {
    let values = [[4, 16], [4, columns]]
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            let id = MidValueId::from_index(index as u32);
            MidValue {
                id,
                tensor_type: TensorType::new(
                    shape,
                    Precision::F32,
                    Layout::row_major(TensorTiling::replicated(1)),
                ),
                origin: ValueId::from_index(index as u32),
                storage_group: id,
                owners: crate::tensor::OwnerMap::default(),
            }
        })
        .collect();
    MidProgram {
        tile_count: 1,
        values,
        inputs: vec![MidInput {
            name: "input".into(),
            kind: GraphInputKind::Host,
            value: MidValueId::from_index(0),
        }],
        outputs: vec![MidValueId::from_index(1)],
        operations: vec![MidOperation {
            source: None,
            inputs: vec![MidValueId::from_index(0)],
            results: vec![MidValueId::from_index(1)],
            kind: MidOperationKind::Copy {
                policy: crate::CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
                mapping: CoordinateMapping::default(),
            },
            operands: Vec::new(),
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
        }],
        ..MidProgram::default()
    }
}

#[test]
fn backed_windows_compose_offsets_without_exposing_mutations() {
    for mutable in [false, true] {
        let mut mid = copied_columns(16);
        mid.values[1].tensor_type.shape.0[0] = 2;
        let mut result = mid.values[1].clone();
        result.id = MidValueId::from_index(2);
        result.storage_group = result.id;
        mid.values.push(result);
        if let MidOperationKind::Copy { mapping, .. } = &mut mid.operations[0].kind {
            mapping.offsets = vec![1, 0];
        }
        let mut consumer = mid.operations[0].clone();
        consumer.kind = MidOperationKind::Gelu;
        consumer.inputs = vec![MidValueId::from_index(1)];
        consumer.results = vec![MidValueId::from_index(2)];
        consumer.operands = vec![OperandIndexing::Fragment(crate::OperandWindow(vec![(
            0, 1, 2,
        )]))];
        mid.operations.push(consumer.clone());
        mid.outputs = consumer.results.clone();
        if mutable {
            let mut alias = mid.values[0].clone();
            alias.id = MidValueId::from_index(3);
            alias.storage_group = alias.id;
            mid.values.push(alias);
            consumer.inputs = vec![MidValueId::from_index(0)];
            consumer.results = vec![MidValueId::from_index(3)];
            consumer.output_aliases = vec![(0, 0)];
            mid.operations.insert(0, consumer);
        }
        mid.use_views();
        if mutable {
            assert!(matches!(
                mid.operations[1].kind,
                MidOperationKind::Copy { .. }
            ));
        } else {
            assert_eq!(mid.operations.len(), 1);
            let OperandIndexing::Fragment(window) = &mid.operations[0].operands[0] else {
                panic!("copy was not replaced with a backed window");
            };
            let source = &mid.values[0].tensor_type;
            let shards = source.format.layout.shard_extents(&source.shape).unwrap();
            let backing = &shards[0].1;
            let selected = window.select(backing, true).unwrap();
            let span = crate::storage::byte_traversal(
                crate::storage::TensorStorage {
                    format: &source.format,
                    extents: backing,
                },
                &selected,
                true,
            )
            .unwrap()
            .contiguous_span()
            .unwrap();
            // Source row 1 was copied; consumer row 1 therefore reads row 2.
            let data = (0..64u32).collect::<Vec<_>>();
            assert_eq!(
                &data[(span.offset / 4) as usize..((span.offset + span.bytes) / 4) as usize],
                &(32..48).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn reduction_fragment_accepts_a_materialized_crop() {
    let mut mid = copied_columns(8);
    for value in &mut mid.values {
        value.tensor_type.format.precision = Precision::F16;
        value.tensor_type.shape.0.insert(0, 2);
        value.tensor_type.format.layout =
            Layout::row_major(TensorTiling::sharded(TensorAxis::FromStart(0), 2));
    }
    mid.tile_count = 2;
    let input = mid.values[1].tensor_type.clone();
    let mut output = input.clone();
    output.shape.0.remove(0);
    output.format.layout = Layout::row_major(TensorTiling::replicated(1));
    let mut builder = crate::planner::fragments::FragmentBuilder::new(&[input]);
    let result = builder
        .sum(
            MidValueId::from_index(0),
            &output,
            0,
            crate::ReductionStaging::Complete,
        )
        .unwrap();
    builder.program.outputs = vec![result];
    builder.program.tile_count = 2;
    mid.outputs = crate::mid::append_fragment(
        &builder.program,
        &[MidValueId::from_index(1)],
        &crate::tensor::OwnerMap::default(),
        None,
        ValueId::from_index(0),
        2,
        &mut mid.values,
        &mut mid.operations,
    )
    .unwrap();
    mid.use_views();
    expand_tiles(&mid, false).unwrap();
}

#[test]
fn grouping_moves_complete_reductions_and_their_preparation() {
    let mut input = TensorType::new(
        [2, 4, 64],
        Precision::F16,
        Layout::row_major(TensorTiling::sharded(TensorAxis::FromStart(0), 2)),
    );
    input.format.layout.order = ElementOrder::Amp(crate::AmpOrder::Left);
    let mut output = input.clone();
    output.shape.0.remove(0);
    output.format.layout.tiling = TensorTiling::replicated(1);
    let mut builder = crate::planner::fragments::FragmentBuilder::new(&[input.clone(), input]);
    let mut groups = Vec::new();
    let mut results = Vec::new();
    for input in 0..2 {
        let begin = builder.program.operations.len();
        let result = builder
            .sum(
                MidValueId::from_index(input),
                &output,
                0,
                crate::ReductionStaging::Complete,
            )
            .unwrap();
        groups.push(
            builder.program.operations[begin..]
                .iter()
                .filter(|op| matches!(op.kind, MidOperationKind::ReductionSum { .. }))
                .map(|op| op.results[0])
                .collect::<Vec<_>>(),
        );
        results.push(result);
    }
    for result in results {
        let id = MidValueId::from_index(builder.program.values.len() as u32);
        let mut value = builder.program.values[result.index() as usize].clone();
        value.id = id;
        value.storage_group = id;
        value.tensor_type.format.layout.order = ElementOrder::RowMajor;
        builder.program.values.push(value);
        builder.program.operations.push(MidOperation {
            source: None,
            inputs: vec![result],
            results: vec![id],
            kind: MidOperationKind::Copy {
                mapping: CoordinateMapping::default(),
                policy: crate::CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
            },
            operands: vec![],
            output_aliases: vec![],
            output_windows: vec![],
        });
        builder.program.outputs.push(id);
    }
    builder.program.tile_count = 4;
    let copies = builder
        .program
        .operations
        .iter()
        .filter(|op| matches!(op.kind, MidOperationKind::Copy { .. }))
        .count();
    builder.program.group_reductions(2).unwrap();
    assert_ne!(
        builder.program.values[groups[0][0].index() as usize].owners,
        builder.program.values[groups[1][0].index() as usize].owners
    );
    assert_eq!(
        copies,
        builder
            .program
            .operations
            .iter()
            .filter(|op| matches!(op.kind, MidOperationKind::Copy { .. }))
            .count()
    );
    for group in groups {
        let final_home = &builder.program.values[group.last().unwrap().index() as usize].owners;
        assert!(
            group
                .iter()
                .all(|id| &builder.program.values[id.index() as usize].owners == final_home),
            "group moved only some panel writers: {group:?}"
        );
    }
    expand_tiles(&builder.program, false).unwrap();
}

#[test]
fn exported_copies_have_complete_storage_and_preserve_values() {
    for columns in [8, 16] {
        let mid = copied_columns(columns);
        let graph = expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let input = low.value_views(low.inputs[0].value)[0].shard;
        let output = low.value_views(low.outputs[0])[0].shard;
        assert_ne!(
            low.shards[output.index() as usize].definition,
            ShardDefinition::Unmaterialized
        );
        assert_eq!(
            low.shards[output.index() as usize].tensor_type.shape.0,
            [4, columns]
        );
        let placement = crate::place(&low).unwrap();
        assert!(placement.shard_addresses.contains_key(&output));
        let source = (0..64u32).collect::<Vec<_>>();
        let actual = {
            let mut actual = vec![u32::MAX; (4 * columns) as usize];
            for copy in &low.local_copies {
                let copy = copy.movement();
                assert_eq!(copy.source, input);
                assert_eq!(copy.destination, output);
                let (rows, bytes, source_stride, destination_stride) = match copy.pattern {
                    CopyPattern::Contiguous => (1, copy.bytes, 0, 0),
                    CopyPattern::Strided {
                        rows,
                        row_bytes,
                        source_stride,
                        destination_stride,
                    } => (rows, row_bytes, source_stride, destination_stride),
                };
                for row in 0..rows {
                    for byte in (0..bytes).step_by(4) {
                        actual[((copy.destination_offset + row * destination_stride + byte) / 4)
                            as usize] = source
                            [((copy.source_offset + row * source_stride + byte) / 4) as usize];
                    }
                }
            }
            actual
        };
        assert_eq!(
            actual,
            (0..4)
                .flat_map(|row| (0..columns).map(move |column| row * 16 + column))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn packed_halfword_sources_are_gathered_before_word_exchange() {
    let mut mid = copied_columns(16);
    mid.tile_count = 4;
    for value in &mut mid.values {
        value.tensor_type.shape.0[0] = 1;
        value.tensor_type.format.precision = Precision::F16;
    }
    let mut source_layout = Layout::amp_left(16, 1);
    source_layout.order = ElementOrder::Amp(AmpOrder::TransposedLeft);
    source_layout.tiling.axes[0] =
        crate::AxisTiling::new(TensorAxis::FromEnd(2), 1, 16, crate::Padding::Zero);
    mid.values[0].tensor_type.format.layout = source_layout;
    mid.values[1].tensor_type.format.layout = Layout::logical_linear(4, 4);
    let graph = expand_tiles(&mid, false).unwrap();
    let low = crate::low::lower_to_tiles(&graph, false);
    let placement = crate::place(&low).unwrap();
    // Each output slice is eight bytes, but the AMP panel interleaves its
    // useful halfwords with padded rows. Raw sends cannot read it.
    let snapshot = crate::exchange::lower_exchanges(
        &low,
        &placement,
        &ipu_target::ipu21::fabric::Topology::c600(),
    )
    .unwrap();
    assert!(
        snapshot
            .iter()
            .any(|phase| phase.activities.iter().any(|tile| !tile.is_empty()))
    );
    let source = low.value_views(low.inputs[0].value)[0].shard;
    assert!(
        low.local_copies
            .iter()
            .any(|copy| copy.movement().source == source && copy.movement().bytes == 2)
    );
    assert!(
        low.local_copies
            .iter()
            .all(|copy| copy.movement().source == source),
        "aligned row-major receivers do not need a second staging copy"
    );
    for phase in &low.exchange_phases {
        for transfer in &phase.transfers {
            assert!(
                view_byte_traversal(
                    &low.shards[transfer.source.shard.index() as usize],
                    &transfer.source,
                    transfer.order,
                )
                .unwrap()
                .word_aligned()
            );
        }
    }
}

#[test]
fn intersection_conversions_read_materialized_crops() {
    let mut mid = copied_columns(8);
    mid.tile_count = 2;
    mid.outputs.clear();
    for axis in [TensorAxis::FromEnd(2), TensorAxis::FromEnd(1)] {
        let mut value = mid.values[1].clone();
        value.id = MidValueId::from_index(mid.values.len() as u32);
        value.storage_group = value.id;
        value.tensor_type.format.layout = Layout::row_major(TensorTiling::sharded(axis, 2));
        mid.operations.push(MidOperation {
            source: None,
            inputs: vec![mid.values[1].id],
            results: vec![value.id],
            kind: MidOperationKind::Copy {
                mapping: CoordinateMapping::default(),
                policy: CopyPolicy::DirectRetile,
                packing: crate::PackingPolicy::Automatic,
            },
            operands: Vec::new(),
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
        });
        mid.outputs.push(value.id);
        mid.values.push(value);
    }
    let graph = expand_tiles(&mid, false).unwrap();
    let low = crate::low::lower_to_tiles(&graph, false);
    let source = low.value_views(low.inputs[0].value)[0].shard;
    let borrowed = &low.value_views(mid.values[1].id)[0];
    assert_ne!(borrowed.shard, source);
    assert_eq!(
        borrowed.extents[1].physical_end - borrowed.extents[1].start,
        8
    );
    assert_eq!(
        low.shards[source.index() as usize].extents[1].physical_end,
        16
    );
    assert!(!low.exchange_phases.is_empty());
    for transfer in low
        .exchange_phases
        .iter()
        .flat_map(|phase| &phase.transfers)
    {
        assert_eq!(transfer.source.shard, borrowed.shard);
        let spans = logical_view_byte_spans(
            &low.shards[borrowed.shard.index() as usize],
            &transfer.source,
        )
        .unwrap();
        // The explicit copy has packed the crop before its consumers exchange it.
        let actual = spans
            .iter()
            .flat_map(|span| span.offset / 4..(span.offset + span.bytes) / 4)
            .collect::<Vec<_>>();
        let bounds = &transfer.source.extents;
        let expected = (bounds[0].start..bounds[0].logical_end)
            .flat_map(|row| {
                (bounds[1].start..bounds[1].logical_end).map(move |column| row * 8 + column)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    crate::place(&low).unwrap();
}

#[test]
fn shifted_halfword_crops_pack_before_physical_exchange() {
    let mut mid = copied_columns(8);
    mid.tile_count = 2;
    for value in &mut mid.values {
        value.tensor_type.format.precision = Precision::F16;
    }
    mid.values[1].tensor_type.format.layout = Layout::row_sharded(2);
    let MidOperationKind::Copy { mapping, .. } = &mut mid.operations[0].kind else {
        unreachable!()
    };
    mapping.offsets = vec![0, 1];
    let graph = expand_tiles(&mid, false).unwrap();
    let low = crate::low::lower_to_tiles(&graph, false);
    let placement = crate::place(&low).unwrap();
    let snapshot = crate::exchange::lower_exchanges(
        &low,
        &placement,
        &ipu_target::ipu21::fabric::Topology::c600(),
    )
    .unwrap();
    assert!(
        snapshot
            .iter()
            .any(|phase| phase.activities.iter().any(|tile| !tile.is_empty()))
    );
    assert!(
        low.local_copies
            .iter()
            .any(|copy| copy.movement().source_offset % 4 == 2)
    );
}

#[test]
fn copied_scalar_keeps_its_semantic_broadcast_shape() {
    let mut mid = copied_columns(1);
    for value in &mut mid.values {
        value.tensor_type.shape.0[0] = 1;
        value.tensor_type.format.precision = Precision::F16;
    }
    let mut result = mid.values[0].clone();
    result.id = MidValueId::from_index(2);
    result.storage_group = result.id;
    mid.outputs = vec![result.id];
    mid.operations.push(MidOperation {
        source: None,
        inputs: vec![mid.values[0].id, mid.values[1].id],
        results: vec![result.id],
        kind: MidOperationKind::Add,
        operands: vec![OperandIndexing::Elementwise { result: 0 }; 2],
        output_aliases: vec![],
        output_windows: Vec::new(),
    });
    mid.values.push(result);
    let graph = expand_tiles(&mid, false).unwrap();
    assert!(!graph.local_copies.is_empty());
    let run = graph
        .kernel_runs
        .iter()
        .find(|run| run.kernel == MidOperationKind::Add)
        .unwrap();
    run.call().unwrap();
    let scalar = &run.inputs[1];
    assert_ne!(
        scalar.shard,
        graph.value_views(graph.inputs[0].value)[0].shard
    );
    assert!(
        scalar
            .extents
            .iter()
            .all(|extent| extent.start == 0 && extent.logical_end == 1)
    );
}

#[test]
fn multi_result_compute_pairs_every_resident_row_with_its_statistics() {
    for (rows, columns, tiles) in [(7, 16, 2), (11, 24, 3), (17, 32, 4)] {
        let data = TensorType::new(
            [rows, columns],
            Precision::F16,
            Layout::logical_linear(tiles, columns),
        );
        let stats = TensorType::new(
            [rows, 1, 2],
            Precision::F32,
            Layout::logical_linear(tiles, 2),
        );
        let values = [data.clone(), data.clone(), stats, data]
            .into_iter()
            .enumerate()
            .map(|(index, tensor_type)| {
                let id = MidValueId::from_index(index as u32);
                MidValue {
                    id,
                    tensor_type,
                    owners: crate::tensor::OwnerMap::default(),
                    origin: ValueId::from_index(index as u32),
                    storage_group: id,
                }
            })
            .collect();
        let mid = MidProgram {
            values,
            tile_count: tiles,
            inputs: (0..2)
                .map(|index| MidInput {
                    name: index.to_string(),
                    kind: GraphInputKind::Host,
                    value: MidValueId::from_index(index),
                })
                .collect(),
            operations: vec![MidOperation {
                source: None,
                inputs: vec![MidValueId::from_index(0), MidValueId::from_index(1)],
                results: vec![MidValueId::from_index(2), MidValueId::from_index(3)],
                kind: MidOperationKind::AddLayerNormMoments,
                operands: vec![OperandIndexing::Elementwise { result: 1 }; 2],
                output_aliases: vec![(1, 0)],
                output_windows: Vec::new(),
            }],
            outputs: vec![MidValueId::from_index(2), MidValueId::from_index(3)],
            ..MidProgram::default()
        };
        mid.validate().unwrap();
        let graph = expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        let mut covered = std::collections::BTreeSet::new();
        for run in low.kernel_calls() {
            if run.kernel != MidOperationKind::AddLayerNormMoments {
                continue;
            }
            crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &Default::default(),
            )
            .unwrap();
            let residual = &run.outputs[1];
            assert!(
                covered.insert(residual.extents[0].start),
                "row computed twice"
            );
            assert_eq!(run.outputs[0].extents[0], residual.extents[0]);
            for operand in &run.inputs {
                assert_eq!(operand.extents, residual.extents);
            }
            assert_eq!(
                placement.shard_addresses[&residual.shard],
                placement.shard_addresses[&run.inputs[0].shard]
            );
        }
        assert_eq!(covered, (0..rows).collect());

        let mut wrong_domain = mid;
        wrong_domain.operations[0].operands[0] = OperandIndexing::Elementwise { result: 0 };
        assert!(
            wrong_domain.validate().is_err(),
            "feature data cannot broadcast into statistics"
        );
    }
}

#[test]
fn writable_aliases_and_reductions_require_complete_copy_buffers() {
    for sum in [false, true] {
        let mut mid = copied_columns(8);
        for value in &mut mid.values {
            value.tensor_type.format.precision = Precision::F16;
            if sum {
                value.tensor_type.shape.0.insert(0, 2);
                value.tensor_type.format.layout =
                    Layout::row_major(TensorTiling::sharded(TensorAxis::FromStart(0), 2));
            }
        }
        mid.tile_count = if sum { 2 } else { 1 };
        let mut result = mid.values[1].clone();
        result.id = MidValueId::from_index(2);
        result.storage_group = result.id;
        if sum {
            result.tensor_type.shape.0.remove(0);
            result.tensor_type.format.layout = Layout::row_major(TensorTiling::replicated(1));
        }
        mid.outputs = vec![mid.values[0].id, result.id];
        if sum {
            let mut builder = crate::planner::fragments::FragmentBuilder { program: mid };
            let output = builder
                .sum(
                    crate::MidValueId::from_index(1),
                    &result.tensor_type,
                    0,
                    crate::ReductionStaging::Complete,
                )
                .unwrap();
            mid = builder.program;
            mid.outputs[1] = output;
        } else {
            mid.operations.push(MidOperation {
                source: None,
                inputs: vec![mid.values[1].id],
                results: vec![result.id],
                kind: MidOperationKind::Gelu,
                operands: vec![OperandIndexing::Elementwise { result: 0 }],
                output_aliases: vec![(0, 0)],
                output_windows: vec![],
            });
            mid.values.push(result);
        }
        let graph = expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let copied = low
            .value_views(mid.values[1].id)
            .iter()
            .map(|view| view.shard)
            .collect::<Vec<_>>();
        for &shard in &copied {
            assert_ne!(
                low.shards[shard.index() as usize].definition,
                ShardDefinition::Unmaterialized
            );
        }
        for run in &low.kernel_runs {
            run.call().unwrap();
        }
        let placement = crate::place(&low).unwrap();
        for &shard in &copied {
            let source = low
                .value_views(low.inputs[0].value)
                .iter()
                .find(|id| {
                    low.shards[id.shard.index() as usize].tile
                        == low.shards[shard.index() as usize].tile
                })
                .unwrap();
            if !sum {
                assert_ne!(
                    placement.shard_addresses[&shard],
                    placement.shard_addresses[&source.shard]
                );
            }
        }
        if !sum {
            let output = low.value_views(low.outputs[1])[0].shard;
            assert_eq!(
                placement.shard_addresses[&output],
                placement.shard_addresses[&copied[0]]
            );
        }
    }
}
