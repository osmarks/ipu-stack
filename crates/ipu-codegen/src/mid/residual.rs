//! Preserve live residuals while fusing their addition with local statistics.
use super::rewrite::{apply_edits, producer_through_copies, same_storage};
use crate::kernel::TileKernelSpec;
use crate::low::CopyPolicy;
use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidValue, MidValueId,
    OperandIndexing,
};
use crate::tensor::{ElementOrder, Padding, Precision, TensorAxis, TensorType};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn fuse(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
) -> bool {
    let mut changed = false;
    for operation in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= fuse(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    let mut removed = BTreeSet::new();
    let mut preparation = BTreeMap::new();
    for index in 0..operations.len() {
        let current = &operations[index];
        let MidOperationKind::Compute(Compute::Kernel {
            kernel, operands, ..
        }) = &current.kind
        else {
            continue;
        };
        let ordinary = *kernel == TileKernelSpec::LayerNorm;
        if (!ordinary && *kernel != TileKernelSpec::LayerNormMoments)
            || operands.iter().any(|indexing| {
                *indexing
                    != if ordinary {
                        OperandIndexing::Elementwise { result: 0 }
                    } else {
                        OperandIndexing::local()
                    }
            })
        {
            continue;
        }
        let Some(&input) = current.inputs.first() else {
            continue;
        };
        let Some((sum, previous, identity_copies)) =
            producer_through_copies(input, &operations[..index], values, !ordinary)
        else {
            continue;
        };
        if removed.contains(&previous) {
            continue;
        }
        let add = &operations[previous];
        let MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Add,
            operands,
            output_aliases,
        }) = &add.kind
        else {
            continue;
        };
        if operands.len() != 2
            || operands
                .iter()
                .any(|indexing| *indexing != (OperandIndexing::Elementwise { result: 0 }))
        {
            continue;
        }
        let value = &values[sum.index() as usize];
        let tensor = &value.tensor_type;
        if tensor.format.precision != Precision::F16
            || tensor.format.layout.order != ElementOrder::RowMajor
            || add.inputs[..2]
                .iter()
                .any(|v| !same_storage(&values[v.index() as usize], value))
        {
            continue;
        }
        // The statistics must describe the version written by this add.
        if operations[previous + 1..index]
            .iter()
            .enumerate()
            .any(|(i, op)| {
                matches!(op.kind, MidOperationKind::Repeat(_))
                    || (!identity_copies.contains(&(previous + 1 + i))
                        && op.results.iter().any(|v| {
                            values[v.index() as usize].storage_group == value.storage_group
                        }))
            })
        {
            continue;
        }
        let redistributed = !same_storage(value, &values[input.index() as usize]);
        let old_value_count = values.len();
        let mut stats_value = None;
        let mut statistic_parts = 1;
        let stats = if ordinary {
            if current.results.len() != 1
                || !same_storage(
                    &values[current.results[0].index() as usize],
                    &values[input.index() as usize],
                )
            {
                continue;
            }
            let Some(stats_type) = row_moments_type(tensor) else {
                continue;
            };
            statistic_parts = *stats_type.shape.0.iter().rev().nth(1).unwrap() as u16;
            let id = MidValueId(values.len() as u32);
            stats_value = Some(MidValue {
                id,
                tensor_type: stats_type,
                storage_group: id,
                ..*value
            });
            id
        } else {
            current.results[0]
        };
        let fused = MidOperation {
            inputs: add.inputs.clone(),
            results: vec![stats, sum],
            kind: MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::AddLayerNormMoments,
                operands: vec![OperandIndexing::Elementwise { result: 1 }; 2],
                output_aliases: output_aliases
                    .iter()
                    .map(|&(_, input)| (1, input))
                    .collect(),
            }),
            ..*add
        };
        let mut apply = current.clone();
        if ordinary {
            apply.inputs[0] = if redistributed { input } else { sum };
            apply.inputs.truncate(3);
            apply.inputs.push(stats);
            apply.kind = MidOperationKind::Compute(Compute::Kernel {
                kernel: TileKernelSpec::LayerNormApply {
                    parts: statistic_parts,
                },
                operands: vec![
                    OperandIndexing::Elementwise { result: 0 },
                    OperandIndexing::Elementwise { result: 0 },
                    OperandIndexing::Elementwise { result: 0 },
                    OperandIndexing::local(),
                ],
                output_aliases: Vec::new(),
            });
        }
        if let Some(value) = stats_value {
            values.push(value);
        }
        let stats_copy = if redistributed {
            let target = &values[input.index() as usize];
            let Some(mut tensor_type) = row_moments_type(&target.tensor_type) else {
                values.truncate(old_value_count);
                continue;
            };
            let rank = target.tensor_type.shape.0.len();
            let target_parts = tensor_type.shape.0[rank - 1] as u16;
            tensor_type.shape.0[rank - 1] = u32::from(statistic_parts);
            tensor_type
                .format
                .layout
                .tiling
                .axes
                .retain(|axis| axis.axis != TensorAxis::FromStart((rank - 1) as u16));
            tensor_type.format.layout.tiling.replicas *= target_parts;
            let id = MidValueId(values.len() as u32);
            let copy = MidOperation {
                source: current.source,
                inputs: vec![stats],
                results: vec![id],
                kind: MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                    policy: CopyPolicy::DirectRetile,
                    packing: crate::PackingPolicy::Automatic,
                },
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            };
            values.push(MidValue {
                id,
                storage_group: id,
                tensor_type,
                ..*target
            });
            *apply.inputs.last_mut().unwrap() = id;
            Some(copy)
        } else {
            None
        };
        if !super::rewrite::fusion_pays(
            "residual/statistics",
            current.source,
            [add, current],
            std::iter::once(&fused)
                .chain(ordinary.then_some(&apply))
                .chain(stats_copy.as_ref()),
            values,
        ) {
            values.truncate(old_value_count);
            continue;
        }
        for copy in identity_copies.into_iter().filter(|_| !redistributed) {
            let result = operations[copy].results[0];
            if super::rewrite::is_single_use(operations, required, result) {
                removed.insert(copy);
            }
        }
        if let Some(copy) = stats_copy {
            preparation.insert(index, vec![copy]);
        }
        operations[previous] = fused;
        if ordinary {
            operations[index] = apply;
        } else {
            removed.insert(index);
        }
        changed = true;
    }
    apply_edits(operations, &removed, preparation);
    changed
}

/// One FP32 mean/variance pair per equal-sized feature partition, on its owners.
fn row_moments_type(tensor: &TensorType) -> Option<TensorType> {
    if tensor.format.layout.order != ElementOrder::RowMajor {
        return None;
    }
    let rank = tensor.shape.0.len();
    let resolved = tensor.format.layout.resolve(&tensor.shape).ok()?;
    let partitions = resolved.axes()?.last()?.partitions();
    let first = partitions.first()?;
    let width = first.logical_end - first.start;
    if width == 0
        || !width.is_multiple_of(4)
        || partitions.iter().any(|part| {
            part.logical_end - part.start != width || part.physical_end != part.logical_end
        })
    {
        return None;
    }
    let mut stats = tensor.clone();
    stats.shape.0.pop()?;
    stats.shape.0.extend([1, 2]);
    stats.format.precision = Precision::F32;
    for axis in &mut stats.format.layout.tiling.axes {
        let index = axis.axis.resolve(rank).ok()?;
        axis.axis = TensorAxis::FromStart(index as u16);
        if index + 1 == rank {
            let parts = u32::from(axis.partitions);
            stats.shape.0[rank - 1] = parts;
            axis.block_size = 1;
            axis.padding_multiple = 1;
            axis.shard_padding_multiple = 1;
            axis.padding_groups = 1;
            axis.padding = Padding::Reject;
        }
    }
    Some(stats)
}

#[cfg(test)]
mod tests {
    use crate::compile::PipelineConfig;
    use crate::estimate::Ipu21CostModel;
    use crate::graph::{ComputeGraph, GraphInputKind, ValueId};
    use crate::mid::{MidInput, MidProgram};
    use crate::planner::test_support::lower;
    use crate::tensor::{AxisTiling, Layout};

    use super::*;

    #[test]
    fn moments_validate_resolved_grouped_partitions() {
        let mut layout = Layout::row_sharded(1);
        layout.tiling.tile_count = 6;
        layout.tiling.axes.push(
            AxisTiling::new(TensorAxis::FromEnd(1), 6, 16, Padding::Zero).with_padding_groups(3),
        );
        for (width, valid) in [(96, true), (144, false), (192, true), (186, false)] {
            // Width 144 has three groups of 48, each split into 32 and 16.
            // Width 186 additionally has logical padding within each group.
            let tensor = TensorType::new([1, width], Precision::F16, layout.clone());
            assert_eq!(row_moments_type(&tensor).is_some(), valid);
        }
    }

    #[test]
    fn residual_fusion_keeps_repeat_carried_results() {
        // Wide rows amortize the extra apply launch. The pipelined standalone
        // add/norm kernels are faster than this fusion for 1152-column rows.
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [4, 9216]).unwrap();
        let r = graph.host_input("residual", [4, 9216]).unwrap();
        let gamma = graph.parameter("gamma", [1, 9216]).unwrap();
        let beta = graph.parameter("beta", [1, 9216]).unwrap();
        let outputs = graph
            .repeat(3, [x, r], [gamma, beta], [], |body, args| {
                let sum = body.add(args.carried[0], args.carried[1])?;
                let norm = body.layer_norm(sum, args.invariants[0], args.invariants[1])?;
                Ok(vec![sum, norm])
            })
            .unwrap();
        graph.set_outputs(outputs).unwrap();
        let config = PipelineConfig::new(4)
            .with_automatic_input(x, Precision::F16)
            .with_automatic_input(r, Precision::F16)
            .with_automatic_input(gamma, Precision::F16)
            .with_automatic_input(beta, Precision::F16);
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let fused = mid
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
        assert!(
            low.kernel_runs
                .iter()
                .any(|run| run.kernel == TileKernelSpec::AddLayerNormMoments)
        );
        assert!(!low.repeat_runs.is_empty());
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        for run in &low.kernel_runs {
            crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &BTreeMap::new(),
            )
            .unwrap();
        }
    }

    #[test]
    fn live_residual_and_statistics_have_independent_allocations() {
        let tensor = TensorType::new([4, 9216], Precision::F16, Layout::row_sharded(4));
        let mut program = MidProgram {
            tile_count: 4,
            ..MidProgram::default()
        };
        for index in 0..6 {
            let id = MidValueId(index);
            program.values.push(MidValue {
                id,
                tile_offset: 0,
                tensor_type: tensor.clone(),
                origin: ValueId::from_index(index),
                storage_group: id,
            });
            if index < 4 {
                program.inputs.push(MidInput {
                    name: index.to_string(),
                    kind: GraphInputKind::Host,
                    value: id,
                });
            }
        }
        for (kernel, inputs, result) in [
            (
                TileKernelSpec::Add,
                vec![MidValueId(0), MidValueId(1)],
                MidValueId(4),
            ),
            (
                TileKernelSpec::LayerNorm,
                vec![MidValueId(4), MidValueId(2), MidValueId(3)],
                MidValueId(5),
            ),
        ] {
            program.operations.push(MidOperation {
                source: None,
                results: vec![result],
                kind: MidOperationKind::Compute(Compute::Kernel {
                    kernel,
                    operands: vec![OperandIndexing::Elementwise { result: 0 }; inputs.len()],
                    output_aliases: Vec::new(),
                }),
                inputs,
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            });
        }
        program.outputs = vec![MidValueId(4), MidValueId(5)];
        let fused = program
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        assert_eq!(fused.outputs, program.outputs);
        assert_eq!(
            fused.operations[0].results,
            vec![MidValueId(6), MidValueId(4)]
        );
        assert_eq!(fused.operations[1].inputs[0], MidValueId(4));
        let mut expanded = (*crate::expand_tiles(&fused).unwrap()).clone();
        for run in &mut expanded.kernel_runs {
            if run.kernel == TileKernelSpec::AddLayerNormMoments {
                std::sync::Arc::make_mut(&mut run.metadata)
                    .requirements
                    .distinct_elements
                    .push(vec![
                        crate::MemoryOperand::Output(0),
                        crate::MemoryOperand::Output(1),
                    ]);
            }
        }
        let low = crate::lower_to_tiles(&std::sync::Arc::new(expanded), false);
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        let mut sums = 0;
        for run in &low.kernel_runs {
            run.call().unwrap();
            let call = crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &BTreeMap::new(),
            )
            .unwrap();
            if run.kernel == TileKernelSpec::AddLayerNormMoments {
                sums += 1;
                assert_eq!(run.outputs.len(), 2);
                let mut occupied = BTreeSet::new();
                for view in &run.outputs {
                    let shard = &low.shards[view.shard.index() as usize];
                    let elements = crate::exchange::effective_memory_elements(
                        placement.shard_addresses[&view.shard],
                        crate::shard_storage_bytes(shard).unwrap().div_ceil(4),
                    );
                    for element in elements {
                        assert!(occupied.insert(element));
                    }
                }
                let mut invalid = run.clone();
                invalid.outputs.pop();
                assert!(invalid.call().is_err());
                let mut invalid = run.clone();
                std::sync::Arc::make_mut(&mut invalid.metadata)
                    .requirements
                    .distinct_elements
                    .push(vec![crate::MemoryOperand::Output(2)]);
                assert!(invalid.call().is_err());
                assert_eq!(call.input_addresses.len(), 3);
                assert_ne!(call.output_address, call.input_addresses[2]);
                assert_eq!(call.arguments, vec![1, 9216]);
                assert!(
                    call.input_addresses[..2]
                        .iter()
                        .all(|a| a != &call.input_addresses[2])
                );
            }
        }
        assert_eq!(sums, 4);

        // Keep the residual on four owners while normalizing on two. Only
        // the tiny statistics follow the existing activation redistribution.
        let mut norm = program.operations.pop().unwrap();
        program.values[5].tensor_type.format.layout = Layout::row_sharded(2);
        for operand in &mut norm.inputs {
            let source = *operand;
            let mut value = program.values[source.index() as usize].clone();
            value.id = MidValueId(program.values.len() as u32);
            value.storage_group = value.id;
            value.tensor_type.format.layout = Layout::row_sharded(2);
            program.operations.push(MidOperation {
                source: None,
                inputs: vec![source],
                results: vec![value.id],
                kind: MidOperationKind::Copy {
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                    policy: CopyPolicy::DirectRetile,
                    packing: crate::PackingPolicy::Automatic,
                },
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            });
            *operand = value.id;
            program.values.push(value);
        }
        let norm_input = norm.inputs[0];
        program.operations.push(norm);
        let fused = program
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .unwrap();
        assert_eq!(fused.operations[0].results[1], MidValueId(4));
        assert_eq!(fused.operations.last().unwrap().inputs[0], norm_input);
        assert_eq!(fused.outputs, program.outputs);
        let graph = crate::expand_tiles(&fused).unwrap();
        for run in &graph.kernel_runs {
            run.call().unwrap();
        }
        assert!(
            graph
                .kernel_runs
                .iter()
                .any(|run| run.kernel == TileKernelSpec::AddLayerNormMoments)
        );
        assert!(
            graph
                .kernel_runs
                .iter()
                .any(|run| run.kernel == TileKernelSpec::LayerNormApply { parts: 1 })
        );

        // The add owns two feature partitions; the apply owns complete rows.
        // Statistics retain an explicit partition axis until their small gather.
        let mut feature_layout = Layout::row_sharded(2);
        feature_layout.tiling.tile_count = 4;
        feature_layout.tiling.axes[0].tile_stride = Some(2);
        feature_layout.tiling.axes.push(
            AxisTiling::new(TensorAxis::FromEnd(1), 2, 4, Padding::Reject).with_tile_stride(1),
        );
        for id in [0, 1, 4] {
            program.values[id].tensor_type.format.layout = feature_layout.clone();
        }
        let fused = program
            .with_elementwise_fusions(
                u64::from(crate::memory::IPU21_DEFAULT_SUPPORT_RESERVATION_BYTES),
                u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES),
            )
            .expect("partial residual statistics should save a scan");
        let low = crate::lower_to_tiles(&crate::expand_tiles(&fused).unwrap(), false);
        let placement = crate::place(&low).unwrap();
        let kernels = crate::KernelBuildPlan::from_program(&low).unwrap();
        let mut applied = 0;
        for run in &low.kernel_runs {
            run.call().unwrap();
            let call = crate::materialize_kernel_run(
                run,
                &low.shards,
                &placement.shard_addresses,
                &kernels,
                &BTreeMap::new(),
            )
            .unwrap();
            if run.kernel == (TileKernelSpec::LayerNormApply { parts: 2 }) {
                assert_eq!(call.arguments, vec![2, 9216, 2]);
                applied += 1;
            }
        }
        assert_eq!(applied, 2);
    }
}
