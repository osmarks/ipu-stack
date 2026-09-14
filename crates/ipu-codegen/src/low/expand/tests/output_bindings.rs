use super::*;
use crate::mid::Compute;
use crate::{
    CoordinateMapping, GraphInputKind, MidInput, MidValue, OperandWindow, TensorAxis, ValueId,
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
                tile_offset: 0,
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
                mapping: CoordinateMapping::default(),
                reuse_local: true,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }],
        ..MidProgram::default()
    }
}

#[test]
fn exported_copies_have_complete_storage_and_preserve_identity_reuse() {
    for columns in [8, 16] {
        let mid = copied_columns(columns);
        let graph = expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let input = low.value_shards(low.inputs[0].value)[0];
        let output = low.value_shards(low.outputs[0])[0];
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
        let actual = if columns == 16 {
            assert!(low.local_copies.is_empty());
            assert_eq!(
                placement.shard_addresses[&output],
                placement.shard_addresses[&input]
            );
            source.clone()
        } else {
            let mut actual = vec![u32::MAX; (4 * columns) as usize];
            for copy in &low.local_copies {
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
    let snapshot = crate::exchange::capture_exchange_schedule(&low, &placement).unwrap();
    assert!(
        snapshot
            .phases
            .iter()
            .any(|phase| !phase.transfers.is_empty())
    );
    let source = low.value_shards(low.inputs[0].value)[0];
    assert!(
        low.local_copies
            .iter()
            .any(|copy| copy.source == source && copy.bytes == 2)
    );
    assert!(
        low.local_copies.iter().all(|copy| copy.source == source),
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
fn intersection_conversions_read_the_backing_storage_of_reused_subviews() {
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
                reuse_local: false,
                policy: CopyPolicy::DirectRetile,
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        });
        mid.outputs.push(value.id);
        mid.values.push(value);
    }
    let graph = expand_tiles(&mid, false).unwrap();
    let low = crate::low::lower_to_tiles(&graph, false);
    let source = low.value_shards(low.inputs[0].value)[0];
    let placeholder = low.value_shards(mid.values[1].id)[0];
    assert_eq!(
        low.shards[placeholder.index() as usize].definition,
        ShardDefinition::Unmaterialized
    );
    assert!(!low.exchange_phases.is_empty());
    for transfer in low
        .exchange_phases
        .iter()
        .flat_map(|phase| &phase.transfers)
    {
        assert_eq!(transfer.source.shard, source);
        let spans = logical_view_byte_spans(&low.shards[source.index() as usize], &transfer.source)
            .unwrap();
        // Every source row still has the original 16-element stride; the
        // borrowed 8-column view does not own a densely packed allocation.
        let actual = spans
            .iter()
            .flat_map(|span| span.offset / 4..(span.offset + span.bytes) / 4)
            .collect::<Vec<_>>();
        let bounds = &transfer.source.extents;
        let expected = (bounds[0].start..bounds[0].logical_end)
            .flat_map(|row| {
                (bounds[1].start..bounds[1].logical_end).map(move |column| row * 16 + column)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    assert!(
        low.local_copies
            .iter()
            .all(|copy| copy.source != placeholder)
    );
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
    let snapshot = crate::exchange::capture_exchange_schedule(&low, &placement).unwrap();
    assert!(
        snapshot
            .phases
            .iter()
            .any(|phase| !phase.transfers.is_empty())
    );
    assert!(
        low.local_copies
            .iter()
            .any(|copy| copy.source_offset % 4 == 2)
    );
}

#[test]
fn borrowed_scalar_keeps_its_semantic_broadcast_shape() {
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
        kind: MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Add,
            operands: vec![OperandWindow::default(); 2],
            output_aliases: vec![],
        }),
        estimated_cycles: 0,
        estimated_exchange_cycles: 0,
    });
    mid.values.push(result);
    let graph = expand_tiles(&mid, false).unwrap();
    assert!(graph.local_copies.is_empty());
    let run = graph
        .kernel_runs
        .iter()
        .find(|run| run.kernel == TileKernelSpec::Add)
        .unwrap();
    crate::validate_kernel_run(run).unwrap();
    let scalar = &run.inputs[1].views[0];
    assert_eq!(scalar.shard, graph.value_shards(graph.inputs[0].value)[0]);
    assert!(
        scalar
            .extents
            .iter()
            .all(|extent| extent.start == 0 && extent.logical_end == 1)
    );
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
        mid.operations.push(MidOperation {
            source: None,
            inputs: vec![mid.values[1].id],
            results: vec![result.id],
            kind: if sum {
                MidOperationKind::Compute(Compute::Sum {
                    axis: 0,
                    staging: crate::ReductionStaging::Complete,
                })
            } else {
                MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::Gelu,
                    operands: vec![OperandWindow::default()],
                    output_aliases: vec![(0, 0)],
                })
            },
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        });
        mid.values.push(result);
        let graph = expand_tiles(&mid, false).unwrap();
        let low = crate::low::lower_to_tiles(&graph, false);
        let copied = low.value_shards(mid.values[1].id);
        for &shard in copied {
            assert_ne!(
                low.shards[shard.index() as usize].definition,
                ShardDefinition::Unmaterialized
            );
        }
        for run in &low.kernel_runs {
            crate::validate_kernel_run(run).unwrap();
        }
        let placement = crate::place(&low).unwrap();
        for &shard in copied {
            let source = low
                .value_shards(low.inputs[0].value)
                .iter()
                .find(|id| {
                    low.shards[id.index() as usize].tile == low.shards[shard.index() as usize].tile
                })
                .unwrap();
            assert_ne!(
                placement.shard_addresses[&shard],
                placement.shard_addresses[source]
            );
        }
        if !sum {
            let output = low.value_shards(low.outputs[1])[0];
            assert_eq!(
                placement.shard_addresses[&output],
                placement.shard_addresses[&copied[0]]
            );
        }
    }
}
