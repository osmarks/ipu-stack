use super::*;
use crate::mid::Compute;
use crate::{GraphInputKind, MidInput, MidValue, OperandIndexing, ValueId};

#[test]
fn primitive_casts_pair_corresponding_linear_fragments() {
    for shape in [[8, 16], [7, 12]] {
        for tiles in [1, 2, 3] {
            let values = [Precision::F32, Precision::F16]
                .into_iter()
                .enumerate()
                .map(|(i, precision)| {
                    let id = MidValueId::from_index(i as u32);
                    MidValue {
                        id,
                        origin: ValueId::from_index(i as u32),
                        storage_group: id,
                        owners: crate::tensor::OwnerMap::default(),
                        tensor_type: TensorType::new(
                            shape,
                            precision,
                            Layout::logical_linear(tiles, 4),
                        ),
                    }
                })
                .collect();
            let input = MidValueId::from_index(0);
            let output = MidValueId::from_index(1);
            let mid = MidProgram {
                tile_count: tiles,
                values,
                inputs: vec![MidInput {
                    name: "input".into(),
                    kind: GraphInputKind::Host,
                    value: input,
                }],
                outputs: vec![output],
                operations: vec![MidOperation {
                    source: None,
                    inputs: vec![input],
                    results: vec![output],
                    kind: MidOperationKind::Compute(Compute::Kernel {
                        kernel: TileKernelSpec::Cast {
                            from: Precision::F32,
                            to: Precision::F16,
                        },
                        operands: vec![OperandIndexing::Elementwise { result: 0 }],
                        output_aliases: vec![],
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                }],
                ..MidProgram::default()
            };
            let expanded = expand_tiles(&mid, false).unwrap();
            assert!(expanded.kernel_runs.len() > usize::from(tiles));
            for run in &expanded.kernel_runs {
                assert_eq!(
                    run.inputs[0].extents, run.outputs[0].extents,
                    "shape {shape:?}, tiles {tiles}"
                );
                run.call().unwrap();
            }
            if shape == [8, 16] && tiles != 3 {
                let mut norm = mid;
                norm.values[0].tensor_type.format.precision = Precision::F16;
                for name in ["gamma", "beta"] {
                    let id = MidValueId::from_index(norm.values.len() as u32);
                    norm.values.push(MidValue {
                        id,
                        origin: ValueId::from_index(id.index()),
                        storage_group: id,
                        owners: crate::tensor::OwnerMap::default(),
                        tensor_type: TensorType::new(
                            [16],
                            Precision::F16,
                            Layout::row_major(TensorTiling::replicated(tiles)),
                        ),
                    });
                    norm.inputs.push(MidInput {
                        name: name.into(),
                        kind: GraphInputKind::Parameter,
                        value: id,
                    });
                    norm.operations[0].inputs.push(id);
                }
                norm.operations[0].kind = MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::LayerNorm,
                    operands: vec![OperandIndexing::local(); 3],
                    output_aliases: vec![],
                });
                let expanded = expand_tiles(&norm, false).unwrap();
                for run in &expanded.kernel_runs {
                    assert_eq!(run.inputs[0].extents, run.outputs[0].extents);
                    run.call().unwrap();
                }
            }
        }
    }
}

#[test]
fn reduction_can_place_its_result_outside_the_partial_owners() {
    use crate::graph::{GraphInputKind, ValueId};
    use crate::mid::{
        Compute, MidInput, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
        ReductionStaging,
    };
    use crate::tensor::{AxisTiling, Layout, OwnerMap, Padding, Precision, TensorAxis, TensorType};
    let mut layout = Layout::row_sharded(3);
    layout.tiling.axes = vec![AxisTiling::new(
        TensorAxis::FromStart(0),
        3,
        1,
        Padding::Reject,
    )];
    let input = TensorType::new([3, 4, 16], Precision::F16, layout);
    let output = TensorType::new([4, 16], Precision::F16, Layout::row_sharded(1));
    let id = MidValueId::from_index;
    let mid = MidProgram {
        tile_count: 8,
        inputs: vec![MidInput {
            name: "partials".into(),
            kind: GraphInputKind::Host,
            value: id(0),
        }],
        outputs: vec![id(1)],
        values: vec![
            MidValue {
                id: id(0),
                storage_group: id(0),
                origin: ValueId::from_index(0),
                tensor_type: input,
                owners: OwnerMap::embedded(vec![0, 2, 4]),
            },
            MidValue {
                id: id(1),
                storage_group: id(1),
                origin: ValueId::from_index(1),
                tensor_type: output,
                owners: OwnerMap::embedded(vec![5]),
            },
        ],
        operations: vec![MidOperation {
            source: None,
            inputs: vec![id(0)],
            results: vec![id(1)],
            kind: MidOperationKind::Compute(Compute::Sum {
                axis: 0,
                staging: ReductionStaging::Complete,
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }],
        ..MidProgram::default()
    };
    mid.validate().unwrap();
    let low = crate::low::expand::expand_tiles(&mid, false).unwrap();
    assert!(
        low.value_shards[1]
            .iter()
            .all(|id| low.shards[id.index() as usize].tile == 5)
    );
    let reductions = low
        .kernel_runs
        .iter()
        .filter(|run| {
            matches!(
                run.kernel,
                crate::kernel::TileKernelSpec::ReductionSum { .. }
            )
        })
        .collect::<Vec<_>>();
    assert!(!reductions.is_empty());
    for run in reductions {
        assert!(
            run.inputs
                .iter()
                .chain(&run.outputs)
                .all(|view| low.shards[view.shard.index() as usize].tile == 5)
        );
    }
    let sent = low
        .exchange_phases
        .iter()
        .flat_map(|phase| &phase.transfers)
        .map(|transfer| low.shards[transfer.source.shard.index() as usize].tile)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(sent, std::collections::BTreeSet::from([0, 2, 4]));
}
