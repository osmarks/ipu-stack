use crate::low::*;
fn lower_to_tiles(
    graph: &crate::MidProgram,
    checkpoints: bool,
) -> super::ExpansionResult<crate::LowProgram> {
    Ok(crate::low::lower_to_tiles(
        &crate::expand_tiles(graph)?,
        checkpoints,
    ))
}
use super::*;
use crate::{
    AccumulationPrecision, AxisTiling, ComputeGraph, ElementOrder, GemmDistribution, GridOrder,
    Ipu21CostModel, Layout, MemoryClass, MidOperator, OperandRequirement, OperatorCandidate,
    OperatorDispatch, Padding, PipelineConfig, Precision, TensorAxis, TensorFormat, TensorTiling,
    TileKernelSpec, lower,
};
use std::collections::BTreeSet;

const CASES: usize = 32;

fn format(tiles: u16) -> TensorFormat {
    TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(tiles),
    }
}

#[test]
fn streamed_conversion_does_not_require_an_adjacent_consumer() {
    let mut graph = ComputeGraph::new();
    let left = graph.host_input("left", [8, 128]).unwrap();
    let right = graph.parameter("right", [128, 64]).unwrap();
    let output = graph.gemm(left, right).unwrap();
    graph.set_outputs([output]).unwrap();
    let input_format = TensorFormat {
        precision: Precision::F32,
        layout: Layout::row_sharded(4),
    };
    let mut config = PipelineConfig::new(4)
        .with_input(left, input_format.clone())
        .with_input(right, input_format);
    config.conversion_streaming = crate::ConversionStreamingPolicy::Always;
    config.operator_candidates.retain(|candidate| {
        matches!(
            candidate.plan.operator,
            MidOperator::Gemm {
                multiply: Precision::F16,
                ..
            }
        )
    });
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let streamed = mid
        .operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            let plan = operation.conversion_plan()?;
            (plan.output.materialization == crate::OperandMaterialization::DispatchSlices)
                .then_some((index, operation.results[0]))
        })
        .collect::<Vec<_>>();
    assert!(
        streamed
            .iter()
            .any(|&(index, result)| !mid.operations[index + 1].inputs.contains(&result))
    );
    let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
    for (_, result) in streamed {
        let value = low.values.iter().find(|value| value.value == result);
        assert!(
            value.is_none_or(|value| value.shards.iter().all(|shard| matches!(
                low.shards[shard.index() as usize].definition,
                ShardDefinition::Unmaterialized
            )))
        );
    }
}

#[test]
fn randomized_linear_shards_cover_flat_storage_once_in_balanced_grains() {
    let mut random = fastrand::Rng::with_seed(0x666c_6174_5f73_6864);
    for case in 0..CASES {
        let rank = random.usize(2..=4);
        let grain = 1_u32 << random.u32(1..=5);
        let mut shape = (0..rank - 1).map(|_| random.u32(1..=5)).collect::<Vec<_>>();
        shape.push(grain * random.u32(1..=8));
        let elements = shape
            .iter()
            .map(|&extent| u64::from(extent))
            .product::<u64>();
        let grains = elements / u64::from(grain);
        let tiles = random.u16(1..=u16::try_from(grains.min(64)).unwrap());
        let tensor = TensorType::new(
            shape.clone(),
            Precision::F16,
            Layout::logical_linear(tiles, grain),
        );
        let shards = tensor.format.layout.shard_extents(&tensor.shape).unwrap();
        let mut coverage = vec![0_u8; usize::try_from(elements).unwrap()];
        let mut tile_elements = vec![0_u64; usize::from(tiles)];
        for (tile, extents) in shards {
            assert_eq!(extents.len(), rank, "case {case}");
            assert!(
                extents[..rank - 1]
                    .iter()
                    .all(|extent| extent.logical_end == extent.start + 1),
                "case {case}"
            );
            let mut row = 0_u64;
            for (axis, extent) in extents[..rank - 1].iter().enumerate() {
                row = row * u64::from(shape[axis]) + u64::from(extent.start);
            }
            let columns = &extents[rank - 1];
            let width = u64::from(shape[rank - 1]);
            for column in columns.start..columns.logical_end {
                let index = usize::try_from(row * width + u64::from(column)).unwrap();
                coverage[index] += 1;
                tile_elements[usize::from(tile)] += 1;
            }
        }
        assert!(coverage.into_iter().all(|count| count == 1), "case {case}");
        assert!(
            tile_elements
                .iter()
                .all(|count| count % u64::from(grain) == 0),
            "case {case}"
        );
        assert!(
            tile_elements.iter().max().unwrap() - tile_elements.iter().min().unwrap()
                <= u64::from(grain),
            "case {case}"
        );
    }
}

#[test]
fn randomized_parallel_reduction_gemms_lower_to_packed_reductions() {
    let mut random = fastrand::Rng::with_seed(0x7472_6565_5f6b_7370);
    for case in 0..CASES {
        let output_columns = [64, 128][random.usize(0..2)];
        let inner_partitions = random.u16(2..=4);
        let column_partitions = random.u16(1..=3);
        let row_partitions = random.u16(inner_partitions..=8);
        let tiles = inner_partitions * column_partitions * row_partitions;
        let rows_per_partition = random.u32(1..=4);
        let rows = u32::from(row_partitions) * rows_per_partition;
        let inner = u32::from(inner_partitions)
            * 64
            * random.u32(1..=u32::from(row_partitions / inner_partitions));
        let columns = u32::from(column_partitions) * output_columns;
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [1, rows, inner]).unwrap();
        let right = graph.parameter("right", [1, inner, columns]).unwrap();
        let product = graph.gemm(left, right).unwrap();
        graph.set_outputs([product]).unwrap();
        let operator = MidOperator::Gemm {
            options: Default::default(),
            multiply: Precision::F16,
            accumulate: AccumulationPrecision::F32,
        };
        let left_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_parallel_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                inner_partitions,
            ),
        };
        let right_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::block_major_matrix_storage(
                64,
                output_columns,
                column_partitions,
                inner_partitions,
                1,
                MemoryClass::Ipu21Interleaved,
            ),
        };
        let (result_row_partitions, result_column_partitions) = if random.bool() {
            (1, 1)
        } else if random.bool() && rows_per_partition >= u32::from(inner_partitions) {
            (inner_partitions, 1)
        } else {
            (1, inner_partitions)
        };
        let storage_rows = row_partitions.saturating_mul(result_row_partitions);
        let storage_columns = column_partitions.saturating_mul(result_column_partitions);
        let reduction_staging = if random.bool() {
            crate::ReductionStaging::Complete
        } else {
            crate::ReductionStaging::Streamed
        };
        let output_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_result_grid(
                if result_column_partitions > 1 {
                    crate::mid::AMP_COLUMN_MICRO
                } else {
                    output_columns
                },
                storage_rows * storage_columns,
                storage_rows,
                storage_columns,
                crate::mid::GridOrder::ColumnsFast,
            ),
        };
        let candidate = OperatorCandidate::new(
            operator,
            [
                OperandRequirement::new(left_format.clone(), 32),
                OperandRequirement::new(right_format.clone(), 32),
            ],
            OperandRequirement::new(output_format, 32),
        )
        .with_dispatch(OperatorDispatch::BlockedGemm {
            inner_block: 64,
            output_column_block: output_columns,
            orientation: crate::GemmOrientation::Normal,
            distribution: GemmDistribution::ParallelReduction {
                row_partitions,
                column_partitions,
                inner_partitions,
                result_row_partitions,
                result_column_partitions,
                reduction_staging,
            },
        });
        let mut config = PipelineConfig::new(tiles)
            .with_input(left, left_format)
            .with_input(right, right_format);
        config.operator_candidates = vec![candidate];
        let mid = lower(&graph, &config, &Ipu21CostModel)
            .unwrap_or_else(|error| panic!("case {case}: {error}"));
        let compact = crate::mid::implementation::resolve(&mid).unwrap();
        let sum = compact
            .operations
            .iter()
            .find(|op| {
                matches!(
                    op.kind,
                    MidOperationKind::Primitive(crate::Primitive::Sum { axis: 0, .. })
                )
            })
            .unwrap();
        assert_eq!(
            compact.values[sum.inputs[0].index() as usize]
                .tensor_type
                .shape
                .0[0],
            u32::from(inner_partitions)
        );
        assert!(
            compact.operations.len() <= 8,
            "distributed GEMM must not enumerate tiles"
        );
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints)
            .unwrap_or_else(|error| {
                panic!(
                    "case {case}: {error}; rows={rows} inner={inner} columns={columns} grid={row_partitions}x{column_partitions}x{inner_partitions}"
                )
            });
        let reduction_runs = low
            .kernel_runs
            .iter()
            .filter(|run| matches!(run.kernel, TileKernelSpec::ReductionSum { .. }))
            .collect::<Vec<_>>();
        assert!(!reduction_runs.is_empty(), "case {case}");
        assert!(
            reduction_runs.iter().all(|run| {
                matches!(
                    run.kernel,
                    TileKernelSpec::ReductionSum { partials }
                        if partials == match reduction_staging {
                            crate::ReductionStaging::Complete => inner_partitions,
                            crate::ReductionStaging::Streamed => 2,
                        }
                ) && run.inputs.len() == 2
            }),
            "case {case}"
        );
        assert!(
            low.exchange_phases.len() <= usize::from(inner_partitions).saturating_add(2),
            "case {case}"
        );
        let parameter_shards = low
            .inputs
            .iter()
            .find(|input| input.kind == crate::GraphInputKind::Parameter)
            .unwrap()
            .shards
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let direct_parameter_runs = low
            .kernel_runs
            .iter()
            .filter(|run| {
                matches!(run.kernel, TileKernelSpec::Gemm { .. })
                    && run.inputs[1]
                        .views
                        .iter()
                        .any(|view| parameter_shards.contains(&view.shard))
            })
            .count();
        assert!(direct_parameter_runs > 0, "case {case}");
        if (result_row_partitions, result_column_partitions) != (1, 1) {
            let output_shards = low.outputs[0]
                .shards
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let packed_results = reduction_runs
                .iter()
                .map(|run| run.output.shard)
                .collect::<BTreeSet<_>>();
            let copied_outputs = low
                .local_copies
                .iter()
                .filter(|copy| packed_results.contains(&copy.source))
                .map(|copy| copy.destination)
                .collect::<BTreeSet<_>>();
            assert!(
                output_shards.is_subset(&copied_outputs),
                "case {case}: every distributed result shard must receive a packed result"
            );
        }
    }
}

#[test]
fn randomized_parameter_owner_groups_pack_independently_of_compute_tiles() {
    let mut random = fastrand::Rng::with_seed(0x7061_7261_6d73);
    for case in 0..CASES {
        let owner_tiles = 1_u16 << random.u32(1..=3);
        let compute_tiles = owner_tiles * 2;
        let inner = u32::from(owner_tiles) * 64;
        let rows = u32::from(compute_tiles) * random.u32(1..=4);
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, inner]).unwrap();
        let right0 = graph.parameter("right.0", [inner, 64]).unwrap();
        let right1 = graph.parameter("right.1", [inner, 64]).unwrap();
        let output0 = graph.gemm(left, right0).unwrap();
        let output1 = graph.gemm(left, right1).unwrap();
        graph.set_outputs([output0, output1]).unwrap();

        let left_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left(64, compute_tiles),
        };
        let right_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::block_major_matrix_storage(
                64,
                64,
                1,
                owner_tiles,
                1,
                MemoryClass::Ipu21Standard,
            ),
        };
        let output_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_result(compute_tiles),
        };
        let mut config = PipelineConfig::new(compute_tiles)
            .with_input(left, left_format.clone())
            .with_input(right0, right_format.clone())
            .with_input(right1, right_format.clone());
        config.operator_candidates = vec![OperatorCandidate::new(
            MidOperator::Gemm {
                options: crate::GemmOptions::default(),
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F16,
            },
            [
                OperandRequirement::new(left_format, 32).with_access_tail(16),
                OperandRequirement::new(right_format, 32),
            ],
            OperandRequirement::new(output_format, 32),
        )];

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        assert!(mid.operations.iter().all(|operation| {
            operation.operator_plan().is_none_or(|plan| {
                plan.requirements.inputs[1].format.layout.tiling.tile_count == owner_tiles
                    && plan.requirements.output.format.layout.tiling.tile_count == compute_tiles
            })
        }));
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        let parameter_tiles = |name: &str| {
            low.inputs
                .iter()
                .find(|input| input.name == name)
                .unwrap()
                .shards
                .iter()
                .map(|shard| low.shards[shard.index() as usize].tile)
                .collect::<BTreeSet<_>>()
        };
        let first = parameter_tiles("right.0");
        let second = parameter_tiles("right.1");
        assert_eq!(first.len(), usize::from(owner_tiles), "case {case}");
        assert_eq!(second.len(), usize::from(owner_tiles), "case {case}");
        assert!(first.is_disjoint(&second), "case {case}");
    }
}

#[test]
fn randomized_pointwise_dispatch_skips_empty_output_shards() {
    let mut random = fastrand::Rng::with_seed(0x656d_7074);
    for case in 0..CASES {
        let tiles = random.u16(2..=32);
        let rows = random.u32(1..u32::from(tiles));
        let columns = random.u32(1..=32) * 2;
        let tensor_format = format(tiles);
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(input, tensor_format.clone());
        config.operator_candidates = vec![OperatorCandidate::new(
            MidOperator::Gelu,
            [OperandRequirement::new(tensor_format.clone(), 8)],
            OperandRequirement::new(tensor_format, 8),
        )];

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        let runs = low
            .tiles
            .iter()
            .flat_map(|tile| low.work(tile))
            .filter_map(|work| match work {
                TileWorkRef::Kernel(run) => Some(run),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(runs.len(), rows as usize, "random case {case}");
        assert!(runs.iter().all(|run| {
            run.output
                .extents
                .iter()
                .all(|extent| extent.start < extent.physical_end)
        }));
    }
}

#[test]
fn randomized_dispatch_streaming_defers_one_use_rearrangements() {
    let mut random = fastrand::Rng::with_seed(0x7374_7265_616d);
    for case in 0..8 {
        let batch = random.u32(1..=4);
        let tokens = 16;
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [batch, tokens, 64]).unwrap();
        let up = graph.parameter("up", [1, 64, 256]).unwrap();
        let down = graph.parameter("down", [1, 256, 64]).unwrap();
        let hidden = graph.gemm(input, up).unwrap();
        let hidden = graph.gelu(hidden).unwrap();
        let output = graph.gemm(hidden, down).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(16)
            .with_active_tile_counts([16])
            .with_automatic_input(input, Precision::F16)
            .with_automatic_input(up, Precision::F16)
            .with_automatic_input(down, Precision::F16);
        config.conversion_streaming = crate::ConversionStreamingPolicy::Always;

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let deferred = mid
            .operations
            .iter()
            .filter_map(|operation| {
                operation.conversion_plan().and_then(|plan| {
                    (plan.output.materialization == crate::OperandMaterialization::DispatchSlices)
                        .then(|| operation.results[0])
                })
            })
            .collect::<BTreeSet<_>>();
        assert!(!deferred.is_empty(), "case {case}");
        let consumers = mid
            .operations
            .iter()
            .filter(|operation| {
                operation
                    .inputs
                    .iter()
                    .any(|input| deferred.contains(input))
            })
            .filter_map(|operation| operation.source)
            .collect::<BTreeSet<_>>();

        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        for run in low
            .tiles
            .iter()
            .flat_map(|tile| low.work(tile))
            .filter_map(|work| match work {
                TileWorkRef::Kernel(run) if matches!(run.kernel, TileKernelSpec::Gemm { .. }) => {
                    Some(run)
                }
                _ => None,
            })
        {
            let output = &low.shards[run.output.shard.index() as usize];
            let flattens_outer_rows = matches!(
                output.tensor_type.format.layout.order,
                ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
            );
            assert!(
                flattens_outer_rows
                    || run.output.extents[..run.output.extents.len() - 2]
                        .iter()
                        .all(|extent| extent.physical_end - extent.start == 1),
                "case {case}"
            );
        }
        assert!(
            low.exchange_phases
                .iter()
                .all(|phase| phase.provenance.reason != WorkReason::LayoutRearrangement),
            "case {case}"
        );
        assert!(
            low.exchange_phases
                .iter()
                .any(|phase| phase.provenance.reason == WorkReason::OperatorInputs),
            "case {case}"
        );
        assert!(
            low.shards
                .iter()
                .filter(|shard| shard.definition == ShardDefinition::Unmaterialized)
                .count()
                >= 16 * deferred.len(),
            "case {case}"
        );
        for run in &low.kernel_runs {
            if run
                .provenance
                .operation
                .is_some_and(|operation| consumers.contains(&operation))
            {
                let input = &run.inputs[0].views[0];
                assert_ne!(
                    low.shards[input.shard.index() as usize].definition,
                    ShardDefinition::Unmaterialized,
                    "case {case}"
                );
                let inner = input.extents.last().unwrap();
                let TileKernelSpec::Gemm { inner_block, .. } = &run.kernel else {
                    continue;
                };
                assert!(
                    inner.physical_end - inner.start <= *inner_block,
                    "case {case}"
                );
            }
        }
    }
}

#[test]
fn randomized_tile_local_gelu_reorders_without_exchange() {
    let mut random = fastrand::Rng::with_seed(0x6765_6c75);
    for case in 0..CASES {
        let row_partitions = 1_u16 << random.u32(0..=3);
        let column_partitions = 1_u16 << random.u32(0..=3);
        let tiles = row_partitions * column_partitions;
        let rows = u32::from(row_partitions) * random.u32(1..=8);
        let columns = u32::from(column_partitions) * 64 * random.u32(1..=4);
        let input_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_output_replicated_grid(tiles, row_partitions, column_partitions),
        };
        let output_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
        };
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(input, input_format.clone());
        config.operator_candidates = vec![OperatorCandidate::new(
            MidOperator::Gelu,
            [OperandRequirement::new(input_format, 8)],
            OperandRequirement::new(output_format, 8),
        )];

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        assert!(low.exchange_phases.is_empty(), "random case {case}");
        for tile in &low.tiles {
            for work in low.work(tile) {
                let TileWorkRef::Kernel(run) = work else {
                    continue;
                };
                assert_eq!(
                    low.shards[run.inputs[0].views[0].shard.index() as usize].tile,
                    tile.tile
                );
                assert_eq!(
                    low.shards[run.output.shard.index() as usize].tile,
                    tile.tile
                );
            }
        }
    }
}

#[test]
fn randomized_same_order_retiles_exchange_into_final_values() {
    let mut random = fastrand::Rng::with_seed(0x6469_7265_6374_7265);
    for case in 0..CASES {
        let source_rows = 1_u16 << random.u32(0..=3);
        let source_columns = 1_u16 << random.u32(0..=3);
        let tiles = source_rows * source_columns;
        let rows = u32::from(source_rows.max(source_columns)) * random.u32(1..=4);
        let columns = u32::from(source_rows.max(source_columns)) * random.u32(1..=4) * 4;
        let layout = |row_partitions, column_partitions| Layout {
            order: ElementOrder::RowMajor,
            tiling: TensorTiling {
                tile_count: tiles,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(TensorAxis::FromEnd(2), row_partitions, 1, Padding::Reject),
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        column_partitions,
                        4,
                        Padding::Reject,
                    ),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        };
        let input_format = TensorFormat {
            precision: Precision::F16,
            layout: layout(source_rows, source_columns),
        };
        let target_format = TensorFormat {
            precision: Precision::F16,
            layout: layout(source_columns, source_rows),
        };
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(input, input_format);
        config.operator_candidates = vec![OperatorCandidate::new(
            MidOperator::Gelu,
            [OperandRequirement::new(target_format.clone(), 8)],
            OperandRequirement::new(target_format, 8),
        )];

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        let conversion_phases = low
            .exchange_phases
            .iter()
            .filter(|phase| phase.provenance.reason == WorkReason::LayoutRearrangement)
            .collect::<Vec<_>>();
        for phase in conversion_phases {
            for destination in phase
                .transfers
                .iter()
                .flat_map(|transfer| &transfer.destinations)
            {
                assert!(matches!(
                    low.shards[destination.shard.index() as usize].definition,
                    ShardDefinition::Value(_)
                ));
            }
        }
        assert!(
            low.shards
                .iter()
                .all(|shard| !matches!(shard.definition, ShardDefinition::ExchangeStaging)),
            "case {case}"
        );
        assert!(low.local_copies.iter().all(|copy| {
            low.shards[copy.source.index() as usize].tile
                == low.shards[copy.destination.index() as usize].tile
        }));
    }
}

#[test]
fn randomized_multiaxis_shards_cover_padded_extents_in_whole_blocks() {
    let mut random = fastrand::Rng::with_seed(0x7368_6172);
    for case in 0..CASES {
        let row_partitions = random.u16(1..=4);
        let column_partitions = random.u16(1..=4);
        let replicas = random.u16(1..=3);
        let row_block = 1_u32 << random.u32(0..=3);
        let column_block = 1_u32 << random.u32(0..=3);
        let tile_count = row_partitions * column_partitions * replicas;
        let layout = Layout {
            order: ElementOrder::RowMajor,
            tiling: TensorTiling {
                tile_count,
                replicas,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        column_partitions,
                        column_block,
                        Padding::Zero,
                    ),
                    AxisTiling::new(
                        TensorAxis::FromEnd(2),
                        row_partitions,
                        row_block,
                        Padding::Zero,
                    ),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        };
        let tensor_type = TensorType::new(
            [
                u32::from(row_partitions) * row_block + random.u32(0..=65),
                u32::from(column_partitions) * column_block + random.u32(0..=65),
            ],
            Precision::F16,
            layout.clone(),
        );
        let padded = layout.padded_shape(&tensor_type.shape).unwrap();
        let shards = tensor_type
            .format
            .layout
            .shard_extents(&tensor_type.shape)
            .unwrap();
        assert_eq!(shards.len(), usize::from(tile_count), "case {case}");

        for (axis, partitions, block) in [
            (0, row_partitions, row_block),
            (1, column_partitions, column_block),
        ] {
            let ranges = shards
                .iter()
                .map(|(_, extents)| (extents[axis].start, extents[axis].physical_end))
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(ranges.len(), usize::from(partitions), "case {case}");
            let mut cursor = 0;
            for (start, end) in ranges {
                assert_eq!(start, cursor, "case {case}");
                assert_eq!(start % block, 0, "case {case}");
                assert_eq!(end % block, 0, "case {case}");
                cursor = end;
            }
            assert_eq!(cursor, padded.0[axis], "case {case}");
        }
    }
}

#[test]
fn randomized_partition_padding_preserves_logical_groups() {
    let mut random = fastrand::Rng::with_seed(0x6772_6f75_705f_7064);
    for case in 0..CASES * 8 {
        let groups = random.u16(1..=16);
        let group_width = random.u32(1..=127);
        let rows = random.u32(1..=16);
        let physical_multiple = 1_u32 << random.u32(1..=4);
        let physical_width = group_width.div_ceil(physical_multiple) * physical_multiple;
        let physical_blocks = physical_width / physical_multiple;
        let partitions_per_group = random.u16(1..=u16::try_from(physical_blocks.min(8)).unwrap());
        let partitions = groups * partitions_per_group;
        let layout = Layout {
            order: ElementOrder::RowMajor,
            tiling: TensorTiling {
                tile_count: partitions,
                replicas: 1,
                axes: vec![
                    AxisTiling::new(
                        TensorAxis::FromEnd(1),
                        partitions,
                        physical_multiple,
                        Padding::Zero,
                    )
                    .with_padding_groups(groups),
                ],
            },
            memory_class: MemoryClass::Ipu21Standard,
        };
        let tensor = TensorType::new(
            [rows, u32::from(groups) * group_width],
            Precision::F16,
            layout,
        );
        let shards = tensor.format.layout.shard_extents(&tensor.shape).unwrap();
        assert_eq!(shards.len(), usize::from(partitions), "case {case}");
        for group in 0..groups {
            let group_base = u32::from(group) * group_width;
            let group_shards = &shards[usize::from(group * partitions_per_group)
                ..usize::from((group + 1) * partitions_per_group)];
            let mut cursor = group_base;
            let mut allocated = 0;
            for (_, extents) in group_shards {
                assert_eq!(extents[1].start, cursor, "case {case}");
                assert!(
                    extents[1].logical_end <= group_base + group_width,
                    "case {case}"
                );
                cursor = extents[1].logical_end;
                allocated += extents[1].physical_end - extents[1].start;
                assert_eq!(
                    crate::shard_storage_bytes(&BlockValue {
                        id: BlockValueId(0),
                        tile: 0,
                        tensor_type: tensor.clone(),
                        extents: extents.clone(),
                        definition: ShardDefinition::Staging,
                    })
                    .unwrap(),
                    rows * (extents[1].physical_end - extents[1].start) * 2,
                    "case {case}"
                );
            }
            assert_eq!(cursor, group_base + group_width, "case {case}");
            assert_eq!(allocated, physical_width, "case {case}");
        }
        assert_eq!(
            crate::estimate::physical_elements(&tensor.shape, &tensor.format.layout),
            u64::from(rows) * u64::from(groups) * u64::from(physical_width),
            "case {case}"
        );
        assert_eq!(
            crate::estimate::maximum_shard_bytes(&tensor),
            u64::from(rows)
                * u64::from(
                    physical_blocks.div_ceil(u32::from(partitions_per_group)) * physical_multiple,
                )
                * 2,
            "case {case}"
        );
    }
}

#[test]
fn randomized_padded_intersections_do_not_claim_adjacent_groups() {
    let mut random = fastrand::Rng::with_seed(0x7064_5f69_6e74_6572);
    for case in 0..CASES * 8 {
        let width = random.u32(1..=127);
        let padding = random.u32(1..=31);
        let group = random.u32(0..=30);
        let start = group * width;
        let owned = ShardExtent {
            axis: 0,
            start,
            logical_end: start + width,
            physical_end: start + width + padding,
        };
        let matching = intersect_extents_with_shared_padding(&[owned], &[owned]).unwrap();
        assert_eq!(matching, vec![owned], "case {case}");

        let adjacent = ShardExtent {
            axis: 0,
            start: start + width,
            logical_end: start + width * 2,
            physical_end: start + width * 2 + padding,
        };
        assert!(
            intersect_extents_with_shared_padding(&[owned], &[adjacent]).is_none(),
            "case {case}"
        );

        let narrower_padding = random.u32(0..=padding);
        let narrower = ShardExtent {
            physical_end: owned.logical_end + narrower_padding,
            ..owned
        };
        let intersection = intersect_extents_with_shared_padding(&[owned], &[narrower]).unwrap();
        assert_eq!(
            intersection[0].physical_end,
            owned.logical_end + narrower_padding,
            "case {case}"
        );
    }
}

#[test]
fn randomized_gemm_grid_orders_align_operands_and_pair_shared_payloads() {
    let mut random = fastrand::Rng::with_seed(0x6772_6964_5f6f_7264);
    for case in 0..CASES {
        let row_partitions = 1_u16 << random.u32(1..=3);
        let column_partitions = 1_u16 << random.u32(1..=3);
        let tiles = row_partitions * column_partitions;
        let rows = u32::from(row_partitions) * random.u32(1..=8);
        let columns = u32::from(column_partitions) * 64 * random.u32(1..=3);
        let inner = 64 * random.u32(1..=4);
        for order in [GridOrder::ColumnsFast, GridOrder::RowsFast] {
            let left = TensorType::new(
                [rows, inner],
                Precision::F16,
                Layout::amp_left_grid(64, tiles, row_partitions, column_partitions, order),
            );
            let right = TensorType::new(
                [inner, columns],
                Precision::F16,
                Layout::block_major_matrix_grid(
                    64,
                    64,
                    tiles,
                    row_partitions,
                    column_partitions,
                    order,
                ),
            );
            let output = TensorType::new(
                [rows, columns],
                Precision::F16,
                Layout::amp_output_grid(64, tiles, row_partitions, column_partitions, order),
            );
            let left = left.format.layout.shard_extents(&left.shape).unwrap();
            let right = right.format.layout.shard_extents(&right.shape).unwrap();
            let output = output.format.layout.shard_extents(&output.shape).unwrap();
            for tile in 0..usize::from(tiles) {
                assert_eq!(left[tile].1[0], output[tile].1[0], "case {case}");
                assert_eq!(right[tile].1[1], output[tile].1[1], "case {case}");
            }
            for tile in (0..usize::from(tiles)).step_by(2) {
                let shared_axis = match order {
                    GridOrder::ColumnsFast => 0,
                    GridOrder::RowsFast => 1,
                };
                assert_eq!(
                    output[tile].1[shared_axis],
                    output[tile + 1].1[shared_axis],
                    "case {case}"
                );
            }
        }
    }
}

#[test]
fn randomized_micro_panel_mappings_carry_word_aligned_row_padding() {
    let mut random = fastrand::Rng::with_seed(0x7061_6464_6564_5f72);
    for case in 0..CASES * 8 {
        let rows = random.u32(1..=AMP_INNER_BLOCK);
        let panel_rows = rows.div_ceil(AMP_COLUMN_MICRO) * AMP_COLUMN_MICRO;
        let source_rows = if random.bool() {
            panel_rows
        } else {
            AMP_INNER_BLOCK
        };
        let source = BlockValue {
            id: BlockValueId(0),
            tile: 0,
            tensor_type: TensorType::new(
                [rows, AMP_COLUMN_MICRO],
                Precision::F16,
                Layout {
                    order: ElementOrder::Amp(AmpOrder::TransposedLeft),
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: source_rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: AMP_COLUMN_MICRO,
                    physical_end: AMP_COLUMN_MICRO,
                },
            ],
            definition: ShardDefinition::ExchangeStaging,
        };
        let destination = BlockValue {
            id: BlockValueId(1),
            tile: 1,
            tensor_type: TensorType::new(
                [rows, AMP_COLUMN_MICRO],
                Precision::F16,
                Layout {
                    order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                        row_block: AMP_INNER_BLOCK as u16,
                        column_block: AMP_COLUMN_MICRO as u16,
                    }),
                    tiling: TensorTiling::replicated(1),
                    memory_class: MemoryClass::Ipu21Standard,
                },
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: AMP_INNER_BLOCK,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: AMP_COLUMN_MICRO,
                    physical_end: AMP_COLUMN_MICRO,
                },
            ],
            definition: ShardDefinition::ExchangeStaging,
        };
        let logical_view = |shard: &BlockValue| ShardView {
            shard: shard.id,
            extents: shard
                .extents
                .iter()
                .copied()
                .map(|mut extent| {
                    extent.physical_end = extent.logical_end;
                    extent
                })
                .collect(),
        };
        let mappings = split_mapping_at_panel_boundaries(
            &source,
            logical_view(&source),
            &destination,
            logical_view(&destination),
        )
        .unwrap_or_else(|error| panic!("case {case}, rows {rows}: {error}"));
        let source_bytes = mappings
            .iter()
            .flat_map(|(view, _)| view_byte_spans(&source, view).unwrap())
            .map(|span| {
                assert_eq!(span.offset & 0b11, 0, "case {case}, rows {rows}");
                assert_eq!(span.bytes & 0b11, 0, "case {case}, rows {rows}");
                span.bytes
            })
            .sum::<u32>();
        let destination_bytes = mappings
            .iter()
            .flat_map(|(_, view)| view_byte_spans(&destination, view).unwrap())
            .map(|span| span.bytes)
            .sum::<u32>();
        assert_eq!(source_bytes, panel_rows * AMP_COLUMN_MICRO * 2);
        assert_eq!(destination_bytes, source_bytes, "case {case}, rows {rows}");
    }
}

#[test]
fn randomized_schedules_make_kernel_operands_resident() {
    let mut random = fastrand::Rng::with_seed(0x6c6f_7721);
    for case in 0..CASES {
        let tiles = 1_u16 << random.u32(0..=3);
        let rows = u32::from(tiles) * random.u32(1..=8) * 16;
        let columns = random.u32(1..=8) * 16;
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, columns]).unwrap();
        let right = graph.host_input("right", [rows, columns]).unwrap();
        let output = graph.add(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(tiles)
            .with_active_tile_counts([tiles])
            .with_input(left, format(tiles))
            .with_input(right, format(tiles));
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        assert_eq!(low.tiles.len(), usize::from(tiles), "case {case}");
        for tile in &low.tiles {
            for work in low.work(tile) {
                if let TileWorkRef::Kernel(run) = work {
                    crate::validate_kernel_run(run).unwrap();
                    assert_eq!(
                        low.shards[run.output.shard.index() as usize].tile,
                        tile.tile
                    );
                    assert!(
                        run.inputs
                            .iter()
                            .flat_map(|operand| &operand.views)
                            .all(|view| {
                                low.shards[view.shard.index() as usize].tile == tile.tile
                            })
                    );
                }
            }
        }
        for phase in &low.exchange_phases {
            assert!(
                low.tiles
                    .iter()
                    .all(|tile| contains_phase(&low, tile, phase.id))
            );
            for transfer in &phase.transfers {
                assert!(transfer.destinations.iter().all(|destination| matches!(
                    low.shards[destination.shard.index() as usize].definition,
                    ShardDefinition::Value(_)
                        | ShardDefinition::ExchangeStaging
                        | ShardDefinition::LocalCopy(_)
                        | ShardDefinition::Staging
                )));
            }
        }
    }
}

#[test]
fn randomized_broadcast_adds_schedule_remote_singleton_views() {
    let mut random = fastrand::Rng::with_seed(0x6272_6463);
    for case in 0..CASES {
        let tiles = 1_u16 << random.u32(1..=3);
        let rows = u32::from(tiles) * random.u32(1..=8);
        let columns = random.u32(1..=8) * 16;
        let mut graph = ComputeGraph::new();
        let bias = graph.host_input("bias", [1, columns]).unwrap();
        let tensor = graph.host_input("tensor", [rows, columns]).unwrap();
        let output = graph.add(bias, tensor).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(tiles)
            .with_input(bias, format(tiles))
            .with_input(tensor, format(tiles));
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        assert!(
            low.exchange_phases
                .iter()
                .any(|phase| matches!(phase.provenance.reason, WorkReason::OperatorInputs)),
            "case {case}"
        );
        for tile in &low.tiles {
            let add = low
                .work(tile)
                .find_map(|work| match work {
                    TileWorkRef::Kernel(run) if matches!(run.kernel, TileKernelSpec::Add) => {
                        Some(run)
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(add.inputs[0].views[0].extents[0].logical_end, 1);
            assert_eq!(
                low.shards[add.inputs[0].views[0].shard.index() as usize].tile,
                tile.tile
            );
        }
    }
}

#[test]
fn randomized_blocked_gemms_expand_to_tile_kernel_phases() {
    let mut random = fastrand::Rng::with_seed(0x6765_6d6d);
    for case in 0..CASES {
        let tiles = 1_u16 << random.u32(0..=3);
        let rows = u32::from(tiles) * random.u32(1..=4) * 8;
        let inner_blocks = random.u32(1..=4);
        let column_blocks = random.u32(1..=4);
        let inner = inner_blocks * 64;
        let columns = column_blocks * 64;
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, inner]).unwrap();
        let right = graph.parameter("right", [inner, columns]).unwrap();
        let output = graph.gemm(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(tiles)
            .with_active_tile_counts([tiles])
            .with_input(left, format(tiles))
            .with_input(right, format(tiles));
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        assert!(std::mem::size_of::<TileWork>() <= 8);
        let mut metadata = Vec::<&Arc<KernelRunMetadata>>::new();
        for run in &low.kernel_runs {
            if let Some(existing) = metadata
                .iter()
                .find(|existing| existing.as_ref() == run.metadata.as_ref())
            {
                assert!(Arc::ptr_eq(existing, &run.metadata), "case {case}");
            } else {
                metadata.push(&run.metadata);
            }
        }

        assert!(low.exchange_phases.iter().all(|phase| {
            phase.provenance.operation.is_some()
                && (phase.provenance.value.is_some()
                    || phase.provenance.reason == WorkReason::OperatorInputs)
        }));
        for tile in &low.tiles {
            let gemms = low
                .work(tile)
                .filter_map(|work| match work {
                    TileWorkRef::Kernel(run)
                        if matches!(run.kernel, TileKernelSpec::Gemm { .. }) =>
                    {
                        Some(run)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut initialized_columns = std::collections::BTreeSet::new();
            for run in gemms {
                assert_eq!(run.provenance.reason, WorkReason::OperatorKernel);
                assert!(run.provenance.operation.is_some());
                assert!(run.provenance.value.is_some());
                let TileKernelSpec::Gemm {
                    mode,
                    inner_block: kernel_inner,
                    output_columns: kernel_columns,
                    ..
                } = run.kernel
                else {
                    unreachable!()
                };
                let output_key = run
                    .output
                    .extents
                    .iter()
                    .map(|extent| (extent.start, extent.physical_end))
                    .collect::<Vec<_>>();
                assert_eq!(
                    mode,
                    if initialized_columns.insert(output_key) {
                        crate::GemmKernelMode::Initialize
                    } else {
                        crate::GemmKernelMode::Accumulate
                    },
                    "case {case}"
                );
                assert_eq!(run.inputs.len(), 2);
                assert!(run.inputs.iter().all(|operand| operand.views.len() == 1));
                assert!(
                    run.inputs[0].views[0]
                        .extents
                        .iter()
                        .any(|extent| { extent.physical_end - extent.start == kernel_inner })
                );
                assert!(
                    run.output
                        .extents
                        .iter()
                        .any(|extent| { extent.physical_end - extent.start == kernel_columns })
                );
            }
        }
    }
}

#[test]
fn randomized_odd_capacities_use_nonempty_active_tile_subsets() {
    let mut random = fastrand::Rng::with_seed(0x7375_6273_6574);
    for case in 0..16 {
        let active_tiles = 1_u16 << random.u32(2..=5);
        let capacity = active_tiles + random.u16(1..active_tiles);
        let rows = u32::from(active_tiles);
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, 64]).unwrap();
        let right = graph.parameter("right", [64, 64]).unwrap();
        let output = graph.gemm(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(capacity)
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let result = mid.operations.last().unwrap().results[0];
        let selected_tiles = mid.values[result.index() as usize]
            .tensor_type
            .format
            .layout
            .tiling
            .tile_count;
        assert!(selected_tiles <= capacity, "case {case}");

        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        assert_eq!(low.tile_count, capacity, "case {case}");
        assert_eq!(low.outputs[0].shards.len(), usize::from(selected_tiles));
        for &shard in &low.outputs[0].shards {
            assert!(
                low.shards[shard.index() as usize]
                    .extents
                    .iter()
                    .all(|extent| extent.start < extent.logical_end),
                "case {case} capacity={capacity} selected={selected_tiles} shard={:?} type={:?}",
                low.shards[shard.index() as usize].extents,
                mid.values[result.index() as usize].tensor_type,
            );
        }
        assert!(low.tiles.iter().all(|tile| tile.tile < capacity));
    }
}

#[test]
fn randomized_resident_blocked_weights_lower_without_panel_copies() {
    let mut random = fastrand::Rng::with_seed(0x7265_7369);
    for _ in 0..48 {
        let tiles = 1_u16 << random.u32(0..=3);
        let rows = u32::from(tiles) * random.u32(1..=4);
        let inner = 64 * random.u32(2..=4);
        let columns = 64 * random.u32(1..=4);
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, inner]).unwrap();
        let right = graph.parameter("right", [inner, columns]).unwrap();
        let output = graph.gemm(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles)
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
        config.operator_candidates.retain(|candidate| {
            matches!(
                candidate.plan.dispatch,
                OperatorDispatch::BlockedGemm {
                    distribution: GemmDistribution::OutputStationary,
                    ..
                }
            ) && candidate
                .plan
                .requirements
                .inputs
                .get(1)
                .is_some_and(|requirement| {
                    requirement.format.layout.order
                        == crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                            row_block: 64,
                            column_block: crate::mid::AMP_COLUMN_MICRO as u16,
                        })
                        && requirement.format.layout.tiling.tile_count == tiles
                        && requirement.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                })
        });
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let operation = mid
            .operations
            .iter()
            .find(|operation| matches!(operation.kind, MidOperationKind::Operator { .. }))
            .unwrap();
        let right_type = &mid.values[operation.inputs[1].index() as usize].tensor_type;
        assert_eq!(
            right_type.format.layout.order,
            crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: crate::mid::AMP_COLUMN_MICRO as u16,
            })
        );
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        assert!(low.tiles.iter().all(|tile| low.work(tile).all(|work| {
            !matches!(work, TileWorkRef::LocalCopy(_)) && !matches!(work, TileWorkRef::Exchange(_))
        })));
        assert!(
            low.tiles
                .iter()
                .flat_map(|tile| low.work(tile))
                .any(|work| {
                    matches!(
                        work,
                        TileWorkRef::Kernel(run)
                            if matches!(run.kernel, TileKernelSpec::Gemm {
                                weights: crate::GemmWeightLoad::Interleaved,
                                ..
                            })
                    )
                })
        );
    }
}

#[test]
fn randomized_partially_sharded_weight_grids_preserve_storage() {
    let mut random = fastrand::Rng::with_seed(0x7374_726d_6765_6d6d);
    for _ in 0..32 {
        let row_partitions = 1_u16 << random.u32(1..=2);
        let inner_partitions = 1_u16 << random.u32(1..=row_partitions.ilog2());
        let column_partitions = 1_u16 << random.u32(0..=2);
        let tiles = row_partitions * column_partitions;
        let rows = u32::from(row_partitions) * random.u32(1..=4);
        let inner_blocks = u32::from(row_partitions) * random.u32(1..=2);
        let inner = inner_blocks * 64;
        let columns = u32::from(column_partitions) * 64;
        let local_staging = if random.bool() {
            crate::LocalOperandStaging::Direct
        } else {
            crate::LocalOperandStaging::MatchRemote
        };
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, inner]).unwrap();
        let right = graph.parameter("right", [inner, columns]).unwrap();
        let output = graph.gemm(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let left_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
        };
        let right_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::block_major_matrix_storage(
                64,
                64,
                column_partitions,
                inner_partitions,
                row_partitions / inner_partitions,
                crate::MemoryClass::Ipu21Standard,
            ),
        };
        let output_format = TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_result_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
        };
        let mut config = PipelineConfig::new(tiles)
            .with_input(left, left_format.clone())
            .with_input(right, right_format.clone());
        config.operator_candidates = vec![crate::OperatorCandidate::new(
            crate::MidOperator::Gemm {
                options: crate::GemmOptions::default(),
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F32,
            },
            [
                crate::OperandRequirement::new(left_format, 32).with_access_tail(16),
                crate::OperandRequirement::new(right_format, 32).with_local_staging(local_staging),
            ],
            crate::OperandRequirement::new(output_format, 32),
        )];
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        let expected_weight_bytes = inner
            .div_ceil(u32::from(inner_partitions))
            .saturating_mul(columns.div_ceil(u32::from(column_partitions)))
            .saturating_mul(2);
        assert!(low.inputs[1].shards.iter().all(|shard| {
            crate::shard_storage_bytes(&low.shards[shard.index() as usize])
                == Ok(expected_weight_bytes)
        }));
    }
}

#[test]
fn randomized_repeats_remain_structured_per_tile() {
    let mut random = fastrand::Rng::with_seed(0x7265_706c);
    for case in 0..CASES {
        let tiles = 1_u16 << random.u32(0..=3);
        let count = random.u32(1..=8);
        let width = u32::from(tiles) * random.u32(1..=8);
        let mut graph = ComputeGraph::new();
        let carried = graph.host_input("carried", [width, 16]).unwrap();
        let parameters = (0..count)
            .map(|index| graph.parameter(format!("parameter.{index}"), [width, 16]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let sequence = graph
            .value_sequence("parameters", parameters.clone())
            .unwrap();
        let result = graph
            .repeat(count, [carried], [], [sequence], |body, arguments| {
                Ok(vec![body.add(arguments.carried[0], arguments.iterated[0])?])
            })
            .unwrap()[0];
        graph.set_outputs([result]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(carried, format(tiles));
        for parameter in parameters {
            config.inputs.insert(parameter, format(tiles));
        }
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        for tile in &low.tiles {
            let repeats = low
                .work(tile)
                .filter_map(|work| match work {
                    TileWorkRef::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(repeats.len(), 1, "case {case}");
            assert_eq!(repeats[0].count, count);
            assert_eq!(repeats[0].iterated[0].inputs.len(), count as usize);
            assert!(repeats[0].iterated[0].stride_bytes > 0);
            assert!(
                repeats[0].iterated[0]
                    .stride_bytes
                    .is_multiple_of(repeats[0].iterated[0].alignment)
            );
            let carried = &repeats[0].carried[0];
            assert_eq!(
                low.shards[carried.argument.index() as usize].definition,
                ShardDefinition::Alias(carried.initial)
            );
            assert_eq!(
                low.shards[carried.yielded.index() as usize].definition,
                ShardDefinition::WritableAlias(carried.argument)
            );
            assert_eq!(
                low.shards[carried.result.index() as usize].definition,
                ShardDefinition::Alias(carried.initial)
            );
            assert!(
                low.work(&repeats[0].body)
                    .any(|work| matches!(work, TileWorkRef::Kernel(_)))
            );
        }
    }
}

#[test]
fn randomized_repeats_alias_fresh_results_after_the_last_carried_use() {
    let mut random = fastrand::Rng::with_seed(0x696e_706c);
    for case in 0..CASES {
        let tiles = 1_u16 << random.u32(0..=3);
        let count = random.u32(1..=4);
        let rows = u32::from(tiles) * random.u32(1..=4) * 8;
        let mut graph = ComputeGraph::new();
        let carried = graph.host_input("carried", [rows, 64]).unwrap();
        let weights = (0..count)
            .map(|index| graph.parameter(format!("weight.{index}"), [64, 64]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
        let result = graph
            .repeat(count, [carried], [], [sequence], |body, arguments| {
                Ok(vec![
                    body.gemm(arguments.carried[0], arguments.iterated[0])?,
                ])
            })
            .unwrap()[0];
        graph.set_outputs([result]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(carried, format(tiles));
        for weight in weights {
            config.inputs.insert(weight, format(tiles));
        }
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        for tile in &low.tiles {
            let repeat = low
                .work(tile)
                .find_map(|work| match work {
                    TileWorkRef::Repeat(repeat) => Some(repeat),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                low.shards[repeat.carried[0].yielded.index() as usize].definition,
                ShardDefinition::WritableAlias(repeat.carried[0].argument),
                "case {case}"
            );
        }
    }
}

fn contains_phase(program: &LowProgram, list: &TileWorkList, phase: ExchangePhaseId) -> bool {
    program.work(list).any(|work| match work {
        TileWorkRef::Exchange(candidate) => candidate == phase,
        TileWorkRef::Repeat(repeat) => contains_phase(program, &repeat.body, phase),
        TileWorkRef::Kernel(_) | TileWorkRef::LocalCopy(_) | TileWorkRef::Checkpoint(..) => false,
    })
}

#[test]
fn general_graph_views_lower_to_correct_relative_copies() {
    for rank in 2..=5 {
        for split in 0..rank {
            for merge in 0..rank {
                if split == merge {
                    continue;
                }
                let mut shape = vec![2; rank];
                shape[split] = 6;
                let mut graph = ComputeGraph::new();
                let input = graph.host_input("input", shape.clone()).unwrap();
                let output = graph
                    .view(input, AxisFactorView::new(split, merge, 3))
                    .unwrap();
                graph.set_outputs([output]).unwrap();
                let config = PipelineConfig::new(1).with_input(
                    input,
                    TensorFormat {
                        precision: Precision::F32,
                        layout: Layout::row_major(TensorTiling::replicated(1)),
                    },
                );
                let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
                let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
                let mut buffers = low
                    .shards
                    .iter()
                    .map(|shard| {
                        vec![u32::MAX; crate::shard_storage_bytes(shard).unwrap() as usize / 4]
                    })
                    .collect::<Vec<_>>();
                let input_shard = low.inputs[0].shards[0].index() as usize;
                buffers[input_shard] = (0..shape.iter().product::<u32>()).collect();
                for work in low.work(&low.tiles[0]) {
                    let copy = match work {
                        TileWorkRef::LocalCopy(copy) => copy,
                        TileWorkRef::Exchange(phase) => {
                            assert!(
                                low.exchange_phases[phase.index() as usize]
                                    .transfers
                                    .is_empty()
                            );
                            continue;
                        }
                        _ => panic!("row-major view should only require copies"),
                    };
                    let (rows, width, source_stride, destination_stride) = match copy.pattern {
                        CopyPattern::Contiguous => (1, copy.bytes, 0, 0),
                        CopyPattern::Strided {
                            rows,
                            row_bytes,
                            source_stride,
                            destination_stride,
                        } => (rows, row_bytes, source_stride, destination_stride),
                    };
                    for row in 0..rows {
                        for byte in (0..width).step_by(4) {
                            let value = buffers[copy.source.index() as usize]
                                [((copy.source_offset + row * source_stride + byte) / 4) as usize];
                            buffers[copy.destination.index() as usize][((copy.destination_offset
                                + row * destination_stride
                                + byte)
                                / 4)
                                as usize] = value;
                        }
                    }
                }
                let output_shape = &graph.value_shape(output).unwrap().0;
                let actual = &buffers[low.outputs[0].shards[0].index() as usize];
                for source in 0..shape.iter().product::<u32>() {
                    let mut index = source;
                    let mut coordinates = vec![0; rank];
                    for axis in (0..rank).rev() {
                        coordinates[axis] = index % shape[axis];
                        index /= shape[axis];
                    }
                    let part = coordinates[split] / output_shape[split];
                    coordinates[split] %= output_shape[split];
                    coordinates[merge] = coordinates[merge] * 3 + part;
                    let target = coordinates
                        .iter()
                        .zip(output_shape)
                        .fold(0, |index, (&coordinate, &width)| index * width + coordinate);
                    assert_eq!(
                        actual[target as usize], source,
                        "rank {rank}, split {split}, merge {merge}"
                    );
                }
            }
        }
    }
}

#[test]
fn physical_micro_panels_preserve_retile_coordinates() {
    let mut rng = fastrand::Rng::with_seed(0x726574696c65);
    for _ in 0..128 {
        let rows = rng.u32(1..=120);
        let columns = rng.u32(2..=20) * 16;
        let make = |id, rows, columns| BlockValue {
            id: BlockValueId(id),
            tile: id as u16,
            tensor_type: TensorType::new(
                [rows, columns],
                Precision::F16,
                Layout::amp_left(columns as u16, 1),
            ),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: rows,
                    physical_end: rows,
                },
                ShardExtent {
                    axis: 1,
                    start: 0,
                    logical_end: columns,
                    physical_end: columns,
                },
            ],
            definition: ShardDefinition::Staging,
        };
        let source = make(0, rows, columns);
        let width = rng.u32(1..columns / 16) * 16;
        let destination = make(1, rows, width);
        let view = ShardView {
            shard: source.id,
            extents: destination.extents.clone(),
        };
        let target = ShardView {
            shard: destination.id,
            extents: destination.extents.clone(),
        };
        let pieces =
            split_mapping_at_panel_boundaries(&source, view.clone(), &destination, target.clone())
                .unwrap();
        let pairs = |a: Vec<ByteSpan>, b: Vec<ByteSpan>| {
            a.into_iter()
                .flat_map(|s| s.offset..s.offset + s.bytes)
                .zip(b.into_iter().flat_map(|s| s.offset..s.offset + s.bytes))
                .collect::<BTreeSet<_>>()
        };
        let expected = pairs(
            logical_view_byte_spans(&source, &view).unwrap(),
            logical_view_byte_spans(&destination, &target).unwrap(),
        );
        assert_eq!(
            pairs(
                view_byte_spans(&source, &view).unwrap(),
                view_byte_spans(&destination, &target).unwrap()
            ),
            expected
        );
        let actual = pieces
            .into_iter()
            .flat_map(|(a, b)| {
                pairs(
                    view_byte_spans(&source, &a).unwrap(),
                    view_byte_spans(&destination, &b).unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
    }
}
