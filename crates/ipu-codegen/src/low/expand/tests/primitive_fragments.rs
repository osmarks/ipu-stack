use super::*;
use crate::mid::MidOperationKind;
use ipu_target::Target;

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
            let mid = MidGraph {
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
                    kind: MidOperationKind::Cast {
                        from: Precision::F32,
                        to: Precision::F16,
                    },
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: vec![],
                    output_windows: Vec::new(),
                }],
                ..MidGraph::default()
            };
            let expanded = expand_tiles(Target::Ipu21, &mid, false).unwrap();
            assert!(expanded.kernel_runs.len() > usize::from(tiles));
            for run in &expanded.kernel_runs {
                assert_eq!(
                    run.inputs[0].extents, run.outputs[0].extents,
                    "shape {shape:?}, tiles {tiles}"
                );
                run.call(Target::Ipu21, None).unwrap();
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
                norm.operations[0].kind = MidOperationKind::LayerNorm;
                norm.operations[0].operands = vec![OperandIndexing::local(); 3];
                norm.operations[0].output_aliases = vec![];
                let expanded = expand_tiles(Target::Ipu21, &norm, false).unwrap();
                for run in &expanded.kernel_runs {
                    assert_eq!(run.inputs[0].extents, run.outputs[0].extents);
                    run.call(Target::Ipu21, None).unwrap();
                }
            }
        }
    }
}
