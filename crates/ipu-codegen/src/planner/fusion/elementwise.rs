//! Fuse compatible executable primitives before physical expansion.
use super::rewrite::{apply_edits, single_use_producers};
use crate::mid::MidOperationKind;

use crate::mid::{MidOperation, MidValue, MidValueId, OperandIndexing};
use crate::tensor::{ElementOrder, Precision, TensorType};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn run(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    let producers = single_use_producers(operations, required);
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let current = &operations[index];
        let crate::MidOperation {
            kind: kernel,
            operands,
            output_aliases: consumer_aliases,
            ..
        } = &current;
        let fused = match kernel {
            MidOperationKind::Gelu => MidOperationKind::BiasGelu,
            MidOperationKind::LayerNorm => MidOperationKind::AddLayerNorm,
            _ => continue,
        };
        if operands
            .iter()
            .any(|indexing| *indexing != (OperandIndexing::Elementwise { result: 0 }))
        {
            continue;
        }
        let Some(&input) = current.inputs.first() else {
            continue;
        };
        let Some(&previous) = producers.get(&input) else {
            continue;
        };
        if previous >= index {
            continue;
        }
        let add = &operations[previous];
        let crate::MidOperation {
            kind: MidOperationKind::Add,
            operands: add_operands,
            output_aliases: add_aliases,
            ..
        } = &add
        else {
            continue;
        };
        if add_operands.len() != 2
            || add_operands
                .iter()
                .any(|indexing| *indexing != (OperandIndexing::Elementwise { result: 0 }))
        {
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
            || left.owners != output_value.owners
            || output_type.format.precision != Precision::F16
            || output_type.format.layout.order != ElementOrder::RowMajor
            || values[input.index() as usize].tensor_type != *output_type
            || values[input.index() as usize].owners != output_value.owners
        {
            continue;
        }
        if !compatible_fusion(&fused, &left.tensor_type, &right.tensor_type, output_type)
            || (fused == MidOperationKind::AddLayerNorm && right.owners != output_value.owners)
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
            !matches!(step.kind, MidOperationKind::Copy { .. })
                || step
                    .results
                    .iter()
                    .chain(step.read_values())
                    .any(|id| groups.contains(&values[id.index() as usize].storage_group))
        }) {
            continue;
        }
        let mut inputs = add.inputs[..2].to_vec();
        inputs.extend_from_slice(&current.inputs[1..operands.len()]);
        let reuse_input = inputs
            .iter()
            .position(|id| {
                let value = &values[id.index() as usize];
                value.storage_group == output_value.storage_group
                    && value.tensor_type == *output_type
            })
            .or_else(|| {
                // Alias edges need not have been folded into storage_group yet.
                // Preserve the proven Add -> consumer alias chain in the replacement.
                consumer_aliases
                    .iter()
                    .any(|&(result, alias)| result == 0 && current.inputs[alias] == input)
                    .then(|| {
                        add_aliases.iter().find_map(|&(result, input)| {
                            let value = &values[add.inputs[input].index() as usize];
                            (result == 0
                                && value.tensor_type == *output_type
                                && value.owners == output_value.owners)
                                .then(|| inputs.iter().position(|id| *id == add.inputs[input]))
                                .flatten()
                        })
                    })
                    .flatten()
            });
        let arity = inputs.len();
        let mut replacement = current.clone();
        replacement.inputs = inputs;
        replacement.kind = fused.clone();
        replacement.operands = vec![OperandIndexing::Elementwise { result: 0 }; arity];
        replacement.output_aliases = reuse_input.map(|input| (0, input)).into_iter().collect();
        if !super::rewrite::fusion_pays(
            if fused == MidOperationKind::BiasGelu {
                "BiasGelu"
            } else {
                "AddLayerNorm"
            },
            current.source,
            [add, current],
            [&replacement],
            values,
        ) {
            continue;
        }
        operations[index] = replacement;
        removed.insert(previous);
        changed = true;
    }
    apply_edits(operations, &removed, BTreeMap::new());
    changed
}

fn compatible_fusion(
    kernel: &MidOperationKind,
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
        MidOperationKind::BiasGelu => {
            right.shape.0.last() == output.shape.0.last()
                && !right.shape.0.is_empty()
                && right.shape.0[..right.shape.0.len() - 1]
                    .iter()
                    .all(|&size| size == 1)
        }
        MidOperationKind::AddLayerNorm => right == output,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::PipelineConfig;
    use crate::estimate::Ipu21CostModel;
    use crate::graph::{ComputeGraph, GraphInputKind, ValueId};
    use crate::low::CopyPolicy;
    use crate::mid::MidProgram;
    use crate::mid::{CoordinateMapping, MidInput};
    use crate::planner::catalogue::OperatorFormatPolicy;
    use crate::planner::operator::OperatorFamily;
    use crate::planner::test_support::lower;
    use crate::tensor::{
        AmpOrder, AxisTiling, Layout, Padding, TensorAxis, TensorFormat, TensorTiling,
    };

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
            let id = MidValueId::from_index(index as u32);
            MidValue {
                id,
                owners: crate::tensor::OwnerMap::default(),
                tensor_type,
                origin: ValueId::from_index(0),
                storage_group: id,
            }
        })
        .collect();
        let operation = |input, result, kind| MidOperation {
            source: None,
            inputs: vec![MidValueId::from_index(input)],
            results: vec![MidValueId::from_index(result)],
            operands: if matches!(kind, MidOperationKind::Copy { .. }) {
                vec![]
            } else {
                vec![OperandIndexing::Elementwise { result: 0 }]
            },
            kind,
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
        };
        let mut program = MidProgram {
            tile_count: 12,
            inputs: vec![MidInput {
                name: "x".into(),
                kind: GraphInputKind::Host,
                value: MidValueId::from_index(0),
            }],
            values,
            operations: vec![
                operation(0, 1, MidOperationKind::Gelu),
                operation(
                    1,
                    2,
                    MidOperationKind::Copy {
                        mapping: CoordinateMapping::default(),
                        policy: CopyPolicy::DirectRetile,
                        packing: crate::PackingPolicy::Automatic,
                    },
                ),
                operation(
                    2,
                    3,
                    MidOperationKind::Cast {
                        from: retiled.format.precision,
                        to: packed.format.precision,
                    },
                ),
            ],
            outputs: vec![MidValueId::from_index(3)],
            ..MidProgram::default()
        };
        let fused = crate::planner::fusion::fuse(&program).unwrap();
        assert_eq!(fused.operations.len(), 2);
        assert_eq!(fused.operations[0].inputs, [MidValueId::from_index(0)]);
        assert_eq!(fused.operations[1].inputs, [MidValueId::from_index(2)]);
        assert!(matches!(fused.operations[1].kind, MidOperationKind::Gelu));
        let graph = crate::expand_tiles(&fused).unwrap();
        for run in &graph.kernel_runs {
            run.call(None).unwrap();
        }
        // Independent arithmetic may intervene; an aliased write may not.
        let mut independent = program.clone();
        let mut extra = independent.values[0].clone();
        extra.id = MidValueId::from_index(independent.values.len() as u32);
        extra.storage_group = extra.id;
        let mut work = independent.operations[0].clone();
        work.results = vec![extra.id];
        independent.outputs.push(extra.id);
        independent.values.push(extra);
        independent.operations.insert(2, work);
        assert!(crate::planner::fusion::fuse(&independent).is_some());
        independent.values.last_mut().unwrap().storage_group = MidValueId::from_index(0);
        assert!(crate::planner::fusion::fuse(&independent).is_none());

        let mut norm = program.clone();
        for _ in 0..2 {
            let id = MidValueId::from_index(norm.values.len() as u32);
            norm.values.push(MidValue {
                id,
                storage_group: id,
                owners: crate::tensor::OwnerMap::default(),
                origin: ValueId::from_index(0),
                tensor_type: TensorType::new(
                    [1152],
                    Precision::F16,
                    Layout::row_major(TensorTiling::replicated(1)),
                ),
            });
            norm.inputs.push(MidInput {
                name: format!("parameter.{}", id.index()),
                kind: GraphInputKind::Parameter,
                value: id,
            });
            norm.operations[0].inputs.push(id);
        }
        norm.operations[0].kind = MidOperationKind::LayerNorm;
        norm.operations[0].operands = vec![OperandIndexing::Elementwise { result: 0 }; 3];
        norm.operations[0].output_aliases = vec![];
        let fused_norm = crate::planner::fusion::fuse(&norm).unwrap();
        assert_eq!(fused_norm.operations.len(), 4); // activation, gamma, beta copies + LN
        let last = fused_norm.operations.last().unwrap();
        assert_eq!(last.inputs.len(), 3);
        assert!(matches!(last.kind, MidOperationKind::LayerNorm));
        let graph = crate::expand_tiles(&fused_norm).unwrap();
        for run in &graph.kernel_runs {
            run.call(None).unwrap();
        }
        let mut bias = norm.clone();
        bias.operations[0].inputs.truncate(2);
        bias.operations[0].kind = MidOperationKind::BiasGelu;
        bias.operations[0].operands = vec![OperandIndexing::Elementwise { result: 0 }; 2];
        bias.operations[0].output_aliases = vec![];
        let fused_bias = crate::planner::fusion::fuse(&bias).unwrap();
        assert_eq!(fused_bias.operations.len(), 3); // activation/bias copies + producer
        let last = fused_bias.operations.last().unwrap();
        assert_eq!(last.inputs.len(), 2);
        assert_ne!(last.inputs[1], bias.operations[0].inputs[1]);
        let graph = crate::expand_tiles(&fused_bias).unwrap();
        for run in &graph.kernel_runs {
            run.call(None).unwrap();
        }
        program.outputs.push(MidValueId::from_index(1));
        assert!(crate::planner::fusion::fuse(&program).is_none());
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
                    let id = MidValueId::from_index(values.len() as u32);
                    values.push(MidValue {
                        id,
                        owners: crate::tensor::OwnerMap::default(),
                        tensor_type: tensor,
                        origin: ValueId::from_index(0),
                        storage_group: id,
                    });
                }
                let inputs = if norm {
                    vec![
                        MidValueId::from_index(0),
                        MidValueId::from_index(1),
                        MidValueId::from_index(2),
                    ]
                } else {
                    vec![MidValueId::from_index(0)]
                };
                let producer = MidOperation {
                    source: None,
                    results: vec![MidValueId::from_index(3)],
                    kind: if norm {
                        MidOperationKind::LayerNorm
                    } else {
                        MidOperationKind::Gelu
                    },
                    operands: vec![OperandIndexing::Elementwise { result: 0 }; inputs.len()],
                    inputs,
                    output_aliases: Vec::new(),
                    output_windows: Vec::new(),
                };
                let mut identity = values[3].clone();
                identity.id = MidValueId::from_index(5);
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
                values.push(identity);
                let copy = MidOperation {
                    source: None,
                    inputs: vec![MidValueId::from_index(3)],
                    results: vec![MidValueId::from_index(5)],
                    kind: MidOperationKind::Copy {
                        mapping: CoordinateMapping::default(),
                        policy: CopyPolicy::DirectRetile,
                        packing: crate::PackingPolicy::Automatic,
                    },
                    operands: Vec::new(),
                    output_aliases: Vec::new(),
                    output_windows: Vec::new(),
                };
                let cast = MidOperation {
                    source: None,
                    inputs: vec![MidValueId::from_index(5)],
                    results: vec![MidValueId::from_index(4)],
                    kind: MidOperationKind::Cast {
                        from: Precision::F16,
                        to: output.format.precision,
                    },
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: Vec::new(),
                    output_windows: Vec::new(),
                };
                let mut program = MidProgram {
                    tile_count: 1,
                    inputs: (0..if norm { 3 } else { 1 })
                        .map(|index| MidInput {
                            name: format!("input.{index}"),
                            kind: GraphInputKind::Host,
                            value: MidValueId::from_index(index),
                        })
                        .collect(),
                    values,
                    operations: vec![producer, copy, cast],
                    outputs: vec![MidValueId::from_index(4)],
                    ..MidProgram::default()
                };
                let fused = crate::planner::fusion::fuse(&program);
                // The faster FP16 affine path makes separate LN + cast
                // cheaper at this width, even for one row.
                assert_eq!(fused.is_some(), !norm);
                if let Some(fused) = fused {
                    let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
                    assert_eq!(low.kernel_runs.len(), 1);
                    low.kernel_runs[0].call(None).unwrap();
                    crate::KernelObjects::from_program(&low).unwrap();
                    let call = low.kernel_runs[0].call(None).unwrap();
                    let mut expected = vec![rows, 1152, (-4i32) as u32, 1];
                    if !norm {
                        expected.extend([1152, 1152]);
                    }
                    assert_eq!(call.arguments, expected);
                }
                program.outputs.push(MidValueId::from_index(3));
                assert!(crate::planner::fusion::fuse(&program).is_none());
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
                // Exercise fusion of compatible whole-row implementations;
                // split-statistics LayerNorm is a separate kernel family.
                config.operator_candidates.retain(|candidate| {
                    !candidate
                        .concrete()
                        .is_some_and(|c| matches!(c.plan.operator, OperatorFamily::LayerNorm))
                        || candidate.format_policy() == OperatorFormatPolicy::RowMajorRows
                });
                if norm {
                    let format = TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::row_major(TensorTiling {
                            tile_count: 4,
                            replicas: 1,
                            axes: vec![
                                AxisTiling::new(TensorAxis::FromEnd(2), 4, 1, Padding::Reject)
                                    .with_tile_stride(1),
                                AxisTiling::new(TensorAxis::FromEnd(3), 1, 1, Padding::Reject)
                                    .with_tile_stride(4),
                                AxisTiling::new(TensorAxis::FromEnd(1), 1, 4, Padding::Reject)
                                    .with_tile_stride(1),
                            ],
                        }),
                    };
                    config = config.with_input(x, format.clone()).with_input(rhs, format);
                }
                let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
                let fused = crate::planner::fusion::fuse(&mid);
                assert_eq!(fused.is_some(), !keep_sum, "norm={norm}");
                if let Some(fused) = fused {
                    if !norm {
                        let (_, before_memory) =
                            crate::estimate::analyze_mid(&mid, &BTreeMap::new()).unwrap();
                        assert!(
                            fused.peak_memory.total <= before_memory.total,
                            "lost the input alias chain"
                        );
                    }
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
                            MidOperationKind::BiasGelu | MidOperationKind::AddLayerNorm
                        ) {
                            run.call(None).unwrap();
                            count += 1;
                        }
                    }
                    assert_eq!(count, 4);
                }
            }
        }
    }
}
