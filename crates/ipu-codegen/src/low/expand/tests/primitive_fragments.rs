use super::*;
use crate::mid::Compute;
use crate::{GraphInputKind, MidInput, MidValue, OperandWindow, ValueId};

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
                        tile_offset: 0,
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
                        operands: vec![OperandWindow::default()],
                        product: None,
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
                    run.inputs[0].views[0].extents, run.output.extents,
                    "shape {shape:?}, tiles {tiles}"
                );
                crate::validate_kernel_run(run).unwrap();
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
                        tile_offset: 0,
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
                    operands: vec![OperandWindow::default(); 3],
                    product: None,
                    output_aliases: vec![],
                });
                let expanded = expand_tiles(&norm, false).unwrap();
                for run in &expanded.kernel_runs {
                    assert_eq!(run.inputs[0].views[0].extents, run.output.extents);
                    crate::validate_kernel_run(run).unwrap();
                }
            }
        }
    }
}
