//! Fuse compatible whole-device primitives before physical expansion.
use super::rewrite::{apply_edits, single_use_producers};
use crate::kernel::TileKernelSpec;
use crate::mid::{
    Compute, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId, OperandIndexing,
};
use crate::tensor::{ElementOrder, Precision, TensorType};
use std::collections::{BTreeMap, BTreeSet};

impl MidProgram {
    pub(crate) fn with_elementwise_fusions(
        &self,
        support_reservation: u64,
        memory_budget: u64,
    ) -> Option<Self> {
        let mut result = self.clone();
        let residual =
            super::residual::fuse(&mut result.operations, &mut result.values, &result.outputs);
        if !fuse_region(&mut result.operations, &mut result.values, &result.outputs) && !residual {
            return None;
        }
        let (before, _) = crate::estimate::analyze_mid(self, &BTreeMap::new())?;
        result.refresh_estimates()?;
        if result.estimated_cycles >= before.total
            || !result
                .peak_memory
                .fits_ipu21_with_budget(support_reservation, memory_budget)
        {
            return None;
        }
        Some(result)
    }
}

fn fuse_region(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= fuse_region(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    let producers = single_use_producers(operations, required);
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let current = &operations[index];
        let MidOperationKind::Compute(Compute::Kernel {
            kernel,
            operands,
            output_aliases: consumer_aliases,
        }) = &current.kind
        else {
            continue;
        };
        let fused = match kernel {
            TileKernelSpec::Gelu => TileKernelSpec::BiasGelu,
            TileKernelSpec::LayerNorm => TileKernelSpec::AddLayerNorm,
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
        let MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Add,
            operands: add_operands,
            output_aliases: add_aliases,
        }) = &add.kind
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
            || (fused == TileKernelSpec::AddLayerNorm && right.owners != output_value.owners)
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
        replacement.kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: fused.clone(),
            operands: vec![OperandIndexing::Elementwise { result: 0 }; arity],
            output_aliases: reuse_input.map(|input| (0, input)).into_iter().collect(),
        });
        if !super::rewrite::fusion_pays(
            if fused == TileKernelSpec::BiasGelu {
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
    changed | super::output_fusion::fuse(operations, values, required)
}

// Shared with the optimistic regional search: known fusion is not a missing kernel.
pub(crate) fn compatible_fusion(
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
    use crate::compile::PipelineConfig;
    use crate::estimate::Ipu21CostModel;
    use crate::graph::{ComputeGraph, GraphInputKind, ValueId};
    use crate::low::CopyPolicy;
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
            let id = MidValueId(index as u32);
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
            inputs: vec![MidValueId(input)],
            results: vec![MidValueId(result)],
            kind,
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        let mut program = MidProgram {
            tile_count: 12,
            inputs: vec![MidInput {
                name: "x".into(),
                kind: GraphInputKind::Host,
                value: MidValueId(0),
            }],
            values,
            operations: vec![
                operation(
                    0,
                    1,
                    MidOperationKind::Compute(Compute::Kernel {
                        kernel: TileKernelSpec::Gelu,
                        operands: vec![OperandIndexing::Elementwise { result: 0 }],
                        output_aliases: vec![],
                    }),
                ),
                operation(
                    1,
                    2,
                    MidOperationKind::Copy {
                        mapping: CoordinateMapping::default(),
                        reuse_local: false,
                        policy: CopyPolicy::DirectRetile,
                        packing: crate::PackingPolicy::Automatic,
                    },
                ),
                operation(
                    2,
                    3,
                    MidOperationKind::Compute(Compute::cast(
                        retiled.format.precision,
                        packed.format.precision,
                    )),
                ),
            ],
            outputs: vec![MidValueId(3)],
            ..MidProgram::default()
        };
        let fused = program
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        assert_eq!(fused.operations.len(), 2);
        assert_eq!(fused.operations[0].inputs, [MidValueId(0)]);
        assert_eq!(fused.operations[1].inputs, [MidValueId(2)]);
        assert!(matches!(
            fused.operations[1].kind,
            MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::Gelu,
                ..
            })
        ));
        let graph = crate::expand_tiles(&fused).unwrap();
        for run in &graph.kernel_runs {
            run.call().unwrap();
        }
        // Independent arithmetic may intervene; an aliased write may not.
        let mut independent = program.clone();
        let mut extra = independent.values[0].clone();
        extra.id = MidValueId(independent.values.len() as u32);
        extra.storage_group = extra.id;
        let mut work = independent.operations[0].clone();
        work.results = vec![extra.id];
        independent.outputs.push(extra.id);
        independent.values.push(extra);
        independent.operations.insert(2, work);
        assert!(
            independent
                .with_elementwise_fusions(
                    u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                    u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
                )
                .is_some()
        );
        independent.values.last_mut().unwrap().storage_group = MidValueId(0);
        assert!(
            independent
                .with_elementwise_fusions(
                    u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                    u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
                )
                .is_none()
        );

        let mut norm = program.clone();
        for _ in 0..2 {
            let id = MidValueId(norm.values.len() as u32);
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
        norm.operations[0].kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::LayerNorm,
            operands: vec![OperandIndexing::Elementwise { result: 0 }; 3],
            output_aliases: vec![],
        });
        let fused_norm = norm
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        assert_eq!(fused_norm.operations.len(), 4); // activation, gamma, beta copies + LN
        let last = fused_norm.operations.last().unwrap();
        assert_eq!(last.inputs.len(), 3);
        assert!(matches!(
            last.kind,
            MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::LayerNorm,
                ..
            })
        ));
        let graph = crate::expand_tiles(&fused_norm).unwrap();
        for run in &graph.kernel_runs {
            run.call().unwrap();
        }
        let mut bias = norm.clone();
        bias.operations[0].inputs.truncate(2);
        bias.operations[0].kind = MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::BiasGelu,
            operands: vec![OperandIndexing::Elementwise { result: 0 }; 2],
            output_aliases: vec![],
        });
        let fused_bias = bias
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        assert_eq!(fused_bias.operations.len(), 3); // activation/bias copies + producer
        let last = fused_bias.operations.last().unwrap();
        assert_eq!(last.inputs.len(), 2);
        assert_ne!(last.inputs[1], bias.operations[0].inputs[1]);
        let graph = crate::expand_tiles(&fused_bias).unwrap();
        for run in &graph.kernel_runs {
            run.call().unwrap();
        }
        program.outputs.push(MidValueId(1));
        assert!(
            program
                .with_elementwise_fusions(
                    u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                    u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
                )
                .is_none()
        );
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
                        owners: crate::tensor::OwnerMap::default(),
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
                    kind: MidOperationKind::Compute(Compute::Kernel {
                        kernel: if norm {
                            TileKernelSpec::LayerNorm
                        } else {
                            TileKernelSpec::Gelu
                        },
                        operands: vec![OperandIndexing::Elementwise { result: 0 }; inputs.len()],
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
                values.push(identity);
                let copy = MidOperation {
                    source: None,
                    inputs: vec![MidValueId(3)],
                    results: vec![MidValueId(5)],
                    kind: MidOperationKind::Copy {
                        mapping: CoordinateMapping::default(),
                        reuse_local: false,
                        policy: CopyPolicy::DirectRetile,
                        packing: crate::PackingPolicy::Automatic,
                    },
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                let cast = MidOperation {
                    source: None,
                    inputs: vec![MidValueId(5)],
                    results: vec![MidValueId(4)],
                    kind: MidOperationKind::Compute(Compute::cast(
                        Precision::F16,
                        output.format.precision,
                    )),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                let mut program = MidProgram {
                    tile_count: 1,
                    inputs: (0..if norm { 3 } else { 1 })
                        .map(|index| MidInput {
                            name: format!("input.{index}"),
                            kind: GraphInputKind::Host,
                            value: MidValueId(index),
                        })
                        .collect(),
                    values,
                    operations: vec![producer, copy, cast],
                    outputs: vec![MidValueId(4)],
                    ..MidProgram::default()
                };
                let fused = program.with_elementwise_fusions(
                    u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                    u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
                );
                // The faster FP16 affine path makes separate LN + cast
                // cheaper at this width, even for one row.
                assert_eq!(fused.is_some(), !norm);
                if let Some(fused) = fused {
                    let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
                    assert_eq!(low.kernel_runs.len(), 1);
                    low.kernel_runs[0].call().unwrap();
                    crate::KernelBuildPlan::from_program(&low).unwrap();
                    let call = low.kernel_runs[0].call().unwrap();
                    let mut expected = vec![rows, 1152, (-4i32) as u32, 1];
                    if !norm {
                        expected.extend([1152, 1152]);
                    }
                    assert_eq!(call.arguments, expected);
                }
                program.outputs.push(MidValueId(3));
                assert!(
                    program
                        .with_elementwise_fusions(
                            u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                            u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES)
                        )
                        .is_none()
                );
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
                let fused = mid.with_elementwise_fusions(
                    u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                    u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
                );
                assert_eq!(fused.is_some(), !keep_sum, "norm={norm}");
                if let Some(fused) = fused {
                    if !norm {
                        let (_, before_memory) =
                            crate::estimate::analyze_mid(&mid, &BTreeMap::new()).unwrap();
                        assert!(
                            fused.peak_memory.total <= before_memory.total,
                            "lost the input alias chain"
                        );
                        // Keeping the consumer output fresh prevents alias chaining;
                        // delaying the bias read can then increase the live peak.
                        let mut fresh = mid.clone();
                        for op in &mut fresh.operations {
                            if let MidOperationKind::Compute(Compute::Kernel {
                                kernel: TileKernelSpec::Gelu,
                                output_aliases,
                                ..
                            }) = &mut op.kind
                            {
                                output_aliases.clear();
                            }
                        }
                        let (_, memory) =
                            crate::estimate::analyze_mid(&fresh, &BTreeMap::new()).unwrap();
                        let unconstrained = fresh
                            .with_elementwise_fusions(
                                config.standard_memory_reservation_bytes,
                                config.tile_memory_budget_bytes,
                            )
                            .unwrap();
                        assert!(unconstrained.peak_memory.total > memory.total);
                        let mut tight = config.clone();
                        tight.standard_memory_reservation_bytes = 0;
                        tight.tile_memory_budget_bytes = memory.total;
                        assert!(memory.fits_ipu21_with_budget(0, tight.tile_memory_budget_bytes));
                        assert!(
                            fresh
                                .with_elementwise_fusions(
                                    tight.standard_memory_reservation_bytes,
                                    tight.tile_memory_budget_bytes
                                )
                                .is_none()
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
                            TileKernelSpec::BiasGelu | TileKernelSpec::AddLayerNorm
                        ) {
                            run.call().unwrap();
                            count += 1;
                        }
                    }
                    assert_eq!(count, 4);
                }
            }
        }
    }
}
