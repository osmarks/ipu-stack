use crate::mid::MidOperationKind;
use crate::planner::catalogue::ConcreteOperatorCandidate;
use crate::tensor::{AMP_INNER_BLOCK, BlockMajorOrder};
fn lower_to_tiles(
    graph: &crate::MidProgram,
    checkpoints: bool,
) -> super::ExpansionResult<crate::LowProgram> {
    let mut graph = graph.clone();
    graph.compose_copies();
    let expanded = super::expand_tiles_cached(&graph, true, Arc::default())?;
    Ok(crate::low::lower_to_tiles(&expanded, checkpoints))
}
use super::*;
use crate::estimate::Ipu21CostModel;
use crate::planner::OperatorCandidate;
use crate::planner::operator::{
    GemmDistribution, OperandRequirement, OperatorDispatch, OperatorFamily,
};
use crate::planner::test_support::lower;
use crate::{
    AccumulationPrecision, AxisTiling, ComputeGraph, ElementOrder, GridOrder, Layout, MemoryClass,
    Padding, PipelineConfig, Precision, TensorAxis, TensorFormat, TensorTiling,
};
use std::collections::BTreeSet;

const CASES: usize = 32;

#[test]
fn exchange_grouping_moves_disjoint_copy_rows_and_preserves_dependencies() {
    for alias in [false, true] {
        for (source, destination, offset, strided, next_source, blocked, hoisted) in [
            (3, 1, 256, false, 1, false, true), // Fill the locally owned tail before exchange.
            (3, 1, 256, true, 1, false, true),
            (3, 1, 0, false, 1, true, false), // Overwrites received bytes; order matters.
            (1, 3, 0, false, 3, true, false), // Actual receive-copy-send dependency.
            (3, 1, 0, false, 3, false, false), // Shared reads permit keeping the copy after.
        ] {
            let mut graph = ComputeGraph::new();
            let input = graph.host_input("input", [16, 16]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            let mid = lower(
                &graph,
                &PipelineConfig::new(4).with_input(input, format(1)),
                &Ipu21CostModel,
            )
            .unwrap();
            let mut builder = TileGraphBuilder::new(&mid, Arc::default()).unwrap();
            let mut ids = vec![builder.shards[0].id];
            for tile in [1, 2, 1] {
                let mut shard = builder.shards[0].clone();
                shard.tile = tile;
                shard.definition = ShardDefinition::Staging;
                ids.push(builder.push_shard(shard).unwrap());
            }
            let mut alias_shard = builder.shards[ids[1].index() as usize].clone();
            alias_shard.definition = ShardDefinition::WritableAlias(ids[1]);
            let alias_id = builder.push_shard(alias_shard).unwrap();
            let mut from = builder.full_view(ids[0]);
            let mut to = builder.full_view(ids[1]);
            for view in [&mut from, &mut to] {
                view.extents[0].logical_end = 8;
                view.extents[0].physical_end = 8;
            }
            let provenance = WorkProvenance {
                operation: Some(graph.operations()[0].id),
                value: None,
                reason: WorkReason::OperatorInputs,
            };
            let mut region = BlockRegion::default();
            builder
                .append_exchange_phase(
                    vec![LogicalExchange {
                        source: from,
                        destinations: vec![to],
                        order: CopyOrder::Physical,
                    }],
                    provenance,
                    &mut region,
                )
                .unwrap();
            builder.local_copies.push(
                crate::kernel::CopyRun::bind(
                    LocalCopy {
                        source: if alias && source == 1 {
                            alias_id
                        } else {
                            ids[source]
                        },
                        destination: if alias && destination == 1 {
                            alias_id
                        } else {
                            ids[destination]
                        },
                        source_offset: 0,
                        destination_offset: offset,
                        bytes: 128,
                        pattern: if strided {
                            CopyPattern::Strided {
                                rows: 4,
                                row_bytes: 32,
                                source_stride: 32,
                                destination_stride: 64,
                            }
                        } else {
                            CopyPattern::Contiguous
                        },
                    },
                    &builder.shards,
                )
                .unwrap(),
            );
            region.operations.push(BlockOperation::Copy {
                tile: 1,
                copy: LocalCopyId(0),
            });
            let second = LogicalExchange {
                source: builder.full_view(ids[next_source]),
                destinations: vec![builder.full_view(ids[2])],
                order: CopyOrder::Physical,
            };
            builder
                .append_exchange_phase(vec![second], provenance, &mut region)
                .unwrap();
            // Appending records the requested sequence; motion is an explicit pass.
            assert_eq!(builder.phases.len(), 2);
            crate::low::passes::movement::group_exchanges(
                &mut region,
                &mut builder.phases,
                &builder.local_copies,
                &builder.shards,
            )
            .unwrap();
            assert_eq!(
                crate::low::passes::movement::group_exchanges(
                    &mut region,
                    &mut builder.phases,
                    &builder.local_copies,
                    &builder.shards
                )
                .unwrap(),
                0
            );
            assert_eq!(
                builder.phases.len(),
                if blocked { 2 } else { 1 },
                "alias={alias}, source={source}, destination={destination}, offset={offset}"
            );
            assert_eq!(
                matches!(region.operations[0], BlockOperation::Copy { .. }),
                hoisted
            );
            if !blocked {
                assert_eq!(builder.phases[0].transfers.len(), 2);
                assert_eq!(builder.phases[0].transfers[0].source.shard, ids[0]);
                assert_eq!(
                    builder.phases[0].transfers[1].source.shard,
                    ids[next_source]
                );
            }
        }
    }
}

#[test]
fn local_materialization_joins_only_compatible_existing_multicasts() {
    for (remote_count, interleaved, exchange_order, loopback) in [
        (2, true, CopyOrder::Physical, true),
        (0, true, CopyOrder::Physical, false),
        (1, true, CopyOrder::Physical, false),
        (2, false, CopyOrder::Physical, true),
        (2, true, CopyOrder::Semantic, false),
    ] {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [16, 16]).unwrap();
        graph.set_outputs([input]).unwrap();
        let config = PipelineConfig::new(3).with_input(input, format(1));
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let mut builder = TileGraphBuilder::new(&mid, Arc::default()).unwrap();
        let source = builder.full_view(builder.shards[0].id);
        let mut mappings = Vec::new();
        for tile in 0..=remote_count {
            let mut destination = builder.shards[0].clone();
            destination.tile = tile;
            destination.definition = ShardDefinition::Staging;
            if interleaved {
                destination.tensor_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
            }
            let id = builder.push_shard(destination).unwrap();
            mappings.push((source.clone(), builder.full_view(id)));
        }
        let provenance = WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::LayoutRearrangement,
        };
        let mut batch = movement::MaterializationBatch::default();
        let mut region = BlockRegion::default();
        builder
            .prepare_mapped_views(
                mappings,
                CopyOrder::Physical,
                exchange_order,
                PackingPolicy::Automatic,
                provenance,
                &mut batch,
                &mut region,
            )
            .unwrap();
        builder
            .append_materialization(batch, provenance, &mut region)
            .unwrap();
        assert_eq!(builder.local_copies.is_empty(), loopback);
        let self_receivers = builder
            .phases
            .iter()
            .flat_map(|phase| &phase.transfers)
            .flat_map(|transfer| &transfer.destinations)
            .filter(|view| builder.shards[view.shard.index() as usize].tile == 0)
            .count();
        assert_eq!(self_receivers, usize::from(loopback));
        if loopback {
            let mut graph = (*crate::expand_tiles(&mid).unwrap()).clone();
            graph.shards = builder.shards;
            graph.exchange_phases = builder.phases;
            graph.local_copies = builder.local_copies;
            graph.kernel_runs = builder.kernel_runs;
            graph.body = region;
            let low = crate::low::lower_to_tiles(&Arc::new(graph), false);
            // Force standard storage into the shared upper region as well.
            let placement = crate::place::place_with_ranges(
                &low,
                &[
                    (
                        ipu_target::ipu21::memory::IPU21_DATA_BASE,
                        ipu_target::ipu21::memory::IPU21_DATA_BASE + 4,
                    ),
                    (
                        ipu_target::ipu21::memory::IPU21_INTERLEAVED_MEMORY_BASE,
                        ipu_target::ipu21::loader_abi::APPLICATION_LOAD_LIMIT,
                    ),
                ],
            )
            .unwrap();
            for transfer in low.exchange_phases.iter().flat_map(|p| &p.transfers) {
                let source = placement.shard_addresses[&transfer.source.shard];
                let source_elements =
                    ipu_target::ipu21::memory::effective_memory_elements(source, 128);
                for destination in &transfer.destinations {
                    if low.shards[destination.shard.index() as usize].tile == 0 {
                        let address = placement.shard_addresses[&destination.shard];
                        assert!(
                            ipu_target::ipu21::memory::effective_memory_elements(address, 128)
                                .iter()
                                .all(|e| !source_elements.contains(e))
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn factor_mappings_keep_the_bound_source_selection() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [1, 4, 32]).unwrap();
    graph.set_outputs([input]).unwrap();
    let config = PipelineConfig::new(1).with_input(input, format(1));
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let mut builder = TileGraphBuilder::new(&mid, Arc::default()).unwrap();
    let source = builder.shards[0].id;
    let source_view = builder
        .narrow_view(&builder.full_view(source), &[(1, 0, 2)])
        .unwrap();
    let source_shape = crate::TensorShape(vec![1, 2, 32]);
    let view = crate::AxisFactorView::new(2, 0, 2);
    let mut destination = builder.shards[0].clone();
    destination.tensor_type.shape = view.output_shape(&source_shape).unwrap();
    destination.extents[0].logical_end = 2;
    destination.extents[0].physical_end = 2;
    destination.extents[1].logical_end = 2;
    destination.extents[1].physical_end = 2;
    destination.extents[2].logical_end = 16;
    destination.extents[2].physical_end = 16;
    let output = builder.push_shard(destination).unwrap();
    let mappings = builder
        .window_view_mappings(&[source_view.clone()], &source_shape, &[output], view, &[])
        .unwrap();
    assert_eq!(mappings.len(), 2);
    // Physical storage is already bound before mapping or copy geometry.
    assert!(
        mappings
            .iter()
            .all(|(input, _)| input.shard == source && input.extents[1].physical_end <= 2)
    );
    let provenance = WorkProvenance {
        operation: None,
        value: None,
        reason: WorkReason::LayoutRearrangement,
    };
    let mut region = BlockRegion::default();
    let mut batch = movement::MaterializationBatch::default();
    builder
        .prepare_mapped_views(
            mappings,
            CopyOrder::Semantic,
            CopyOrder::Semantic,
            PackingPolicy::Automatic,
            provenance,
            &mut batch,
            &mut region,
        )
        .unwrap();
    builder
        .append_materialization(batch, provenance, &mut region)
        .unwrap();
    assert!(!builder.local_copies.is_empty());
    assert!(
        builder
            .local_copies
            .iter()
            .all(|copy| copy.movement().source == source)
    );

    let run = builder
        .bind_kernel(
            provenance,
            MidOperationKind::Gelu,
            vec![source_view.clone()],
            vec![source_view.clone()],
        )
        .unwrap();
    assert_eq!(run.inputs[0], source_view);
    let mut outside = source_view;
    outside.extents[2].physical_end += 1;
    assert!(
        builder
            .bind_kernel(
                provenance,
                MidOperationKind::Gelu,
                vec![outside],
                vec![run.outputs[0].clone()]
            )
            .is_err()
    );
    let mut unavailable = builder.shards[source.index() as usize].clone();
    unavailable.definition = ShardDefinition::Unmaterialized;
    let unavailable = builder.push_shard(unavailable).unwrap();
    assert!(matches!(
        builder.bind_kernel(
            provenance,
            MidOperationKind::Gelu,
            vec![builder.full_view(unavailable)],
            vec![builder.full_view(source)]
        ),
        Err(ExpansionError::Kernel(crate::KernelError::Storage(
            StorageError::InvalidView
        )))
    ));
}

fn format(tiles: u16) -> TensorFormat {
    TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(tiles),
    }
}

#[test]
fn panel_construction_keeps_both_operand_casts_materialized() {
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
            candidate.operator(),
            OperatorFamily::Gemm {
                multiply: Precision::F16,
                ..
            }
        )
    });
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let casts = mid
        .operations
        .iter()
        .filter(|op| matches!(op.kind, MidOperationKind::Cast { .. }))
        .collect::<Vec<_>>();
    assert_eq!(casts.len(), 2);
    let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
    for cast in casts {
        assert!(!low.value_views(cast.results[0]).is_empty());
        assert!(
            low.value_views(cast.results[0])
                .iter()
                .all(|shard| !matches!(
                    low.shards[shard.shard.index() as usize].definition,
                    ShardDefinition::Unmaterialized
                ))
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
        let operator = OperatorFamily::Gemm {
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
                    crate::tensor::AMP_COLUMN_MICRO
                } else {
                    output_columns
                },
                storage_rows * storage_columns,
                storage_rows,
                storage_columns,
                crate::tensor::GridOrder::ColumnsFast,
            ),
        };
        let candidate = ConcreteOperatorCandidate::new(
            operator,
            [
                OperandRequirement::new(left_format.clone()),
                OperandRequirement::new(right_format.clone()),
            ],
            output_format,
        )
        .with_dispatch(OperatorDispatch::BlockedGemm {
            inner_block: 64,
            output_column_block: output_columns,
            orientation: crate::planner::operator::GemmOrientation::Normal,
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
        config.operator_candidates = vec![candidate]
            .into_iter()
            .map(OperatorCandidate::Concrete)
            .collect();
        let mid = lower(&graph, &config, &Ipu21CostModel)
            .unwrap_or_else(|error| panic!("case {case}: {error}"));
        let compact = mid.clone();
        let sum = compact
            .operations
            .iter()
            .find(|op| matches!(op.kind, MidOperationKind::Sum { axis: 0, .. }))
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
            .filter(|run| matches!(run.kernel, MidOperationKind::ReductionSum { .. }))
            .collect::<Vec<_>>();
        assert!(!reduction_runs.is_empty(), "case {case}");
        assert!(
            reduction_runs.iter().all(|run| {
                matches!(
                    run.kernel,
                    MidOperationKind::ReductionSum { partials }
                        if partials == match reduction_staging {
                            crate::ReductionStaging::Complete => inner_partitions,
                            crate::ReductionStaging::Streamed => 2,
                            crate::ReductionStaging::Batched(limit) => limit.get() + 1,
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
            .value_views(
                low.inputs
                    .iter()
                    .find(|input| input.kind == crate::GraphInputKind::Parameter)
                    .unwrap()
                    .value,
            )
            .iter()
            .map(|view| view.shard)
            .collect::<BTreeSet<_>>();
        let direct_parameter_runs = low
            .kernel_runs
            .iter()
            .filter(|run| {
                matches!(run.kernel, MidOperationKind::Gemm { .. })
                    && parameter_shards.contains(&run.inputs[1].shard)
            })
            .count();
        assert!(direct_parameter_runs > 0, "case {case}");
        if (result_row_partitions, result_column_partitions) != (1, 1) {
            let output_shards = low
                .value_views(low.outputs[0])
                .iter()
                .map(|view| view.shard)
                .collect::<BTreeSet<_>>();
            let packed_results = reduction_runs
                .iter()
                .map(|run| run.outputs[0].shard)
                .collect::<BTreeSet<_>>();
            let copied_outputs = low
                .local_copies
                .iter()
                .filter(|copy| packed_results.contains(&copy.movement().source))
                .map(|copy| copy.movement().destination)
                .collect::<BTreeSet<_>>();
            assert!(
                output_shards.iter().all(
                    |output| copied_outputs.contains(output) || packed_results.contains(output)
                ),
                "case {case}: every distributed result shard must be written by a reduction or its copy"
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
        config.operator_candidates = vec![ConcreteOperatorCandidate::new(
            OperatorFamily::Gemm {
                options: crate::GemmOptions::default(),
                multiply: Precision::F16,
                accumulate: crate::AccumulationPrecision::F16,
            },
            [
                OperandRequirement::new(left_format),
                OperandRequirement::new(right_format),
            ],
            output_format,
        )]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        let parameter_tiles = |name: &str| {
            low.value_views(
                low.inputs
                    .iter()
                    .find(|input| input.name == name)
                    .unwrap()
                    .value,
            )
            .iter()
            .map(|shard| low.shards[shard.shard.index() as usize].tile)
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
        config.operator_candidates = vec![ConcreteOperatorCandidate::new(
            OperatorFamily::Gelu,
            [OperandRequirement::new(tensor_format.clone())],
            tensor_format,
        )]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        let runs = low
            .tiles
            .iter()
            .flat_map(|tile| tile.work.iter())
            .filter_map(|work| match work {
                BlockOperation::Compute { run, .. } => Some(&low.kernel_runs[run.0 as usize]),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(runs.len(), rows as usize, "random case {case}");
        assert!(runs.iter().all(|run| {
            run.outputs[0]
                .extents
                .iter()
                .all(|extent| extent.start < extent.physical_end)
        }));
    }
}

#[test]
fn randomized_panel_consumers_have_bounded_materialized_operands() {
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
            .with_input(
                input,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::row_sharded(16),
                },
            )
            .with_automatic_input(up, Precision::F16)
            .with_automatic_input(down, Precision::F16);
        config.conversion_streaming = crate::ConversionStreamingPolicy::Always;

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        mid.validate().unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        for run in low
            .tiles
            .iter()
            .flat_map(|tile| tile.work.iter())
            .filter_map(|work| match work {
                BlockOperation::Compute { run, .. }
                    if matches!(
                        low.kernel_runs[run.0 as usize].kernel,
                        MidOperationKind::Gemm { .. }
                    ) =>
                {
                    Some(&low.kernel_runs[run.0 as usize])
                }
                _ => None,
            })
        {
            let output = &low.shards[run.outputs[0].shard.index() as usize];
            let flattens_outer_rows = matches!(
                output.tensor_type.format.layout.order,
                ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
            );
            assert!(
                flattens_outer_rows
                    || run.outputs[0].extents[..run.outputs[0].extents.len() - 2]
                        .iter()
                        .all(|extent| extent.physical_end - extent.start == 1),
                "case {case}"
            );
        }
        for run in &low.kernel_runs {
            if matches!(run.kernel, MidOperationKind::Gemm { .. }) {
                let input = &run.inputs[0];
                assert_ne!(
                    low.shards[input.shard.index() as usize].definition,
                    ShardDefinition::Unmaterialized,
                    "case {case}"
                );
                let inner = input.extents.last().unwrap();
                let MidOperationKind::Gemm { inner_block, .. } = &run.kernel else {
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
fn randomized_tile_local_gelu_conversions_do_not_require_exchange() {
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
                crate::tensor::GridOrder::ColumnsFast,
            ),
        };
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(input, input_format.clone());
        config.operator_candidates = vec![ConcreteOperatorCandidate::new(
            OperatorFamily::Gelu,
            // GeLU preserves element order. Requesting its output format on
            // the operand makes the required local conversion explicit.
            [OperandRequirement::new(output_format.clone())],
            output_format,
        )]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();

        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        assert!(low.exchange_phases.is_empty(), "random case {case}");
        for tile in &low.tiles {
            for work in tile.work.iter() {
                let BlockOperation::Compute { run, .. } = work else {
                    continue;
                };
                let run = &low.kernel_runs[run.0 as usize];
                assert_eq!(
                    low.shards[run.inputs[0].shard.index() as usize].tile,
                    tile.tile
                );
                assert_eq!(
                    low.shards[run.outputs[0].shard.index() as usize].tile,
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
        config.operator_candidates = vec![ConcreteOperatorCandidate::new(
            OperatorFamily::Gelu,
            [OperandRequirement::new(target_format.clone())],
            target_format,
        )]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();

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
            low.shards[copy.movement().source.index() as usize].tile
                == low.shards[copy.movement().destination.index() as usize].tile
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
        let (precision, micro) = if case % 2 == 0 {
            (Precision::F16, 16)
        } else {
            (Precision::F8F143 { scale_exponent: -4 }, 32)
        };
        let rows = random.u32(1..=AMP_INNER_BLOCK);
        let panel_rows = rows.div_ceil(micro) * micro;
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
                precision,
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
                precision,
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
        // Input mappings may already carry shared tail padding. Splitting
        // them again must not extend either side beyond its allocation.
        let mut padded_source = logical_view(&source);
        let mut padded_destination = logical_view(&destination);
        padded_source.extents[0].physical_end = panel_rows;
        padded_destination.extents[0].physical_end = panel_rows;
        assert_eq!(
            mappings,
            split_mapping_at_panel_boundaries(
                &source,
                padded_source,
                &destination,
                padded_destination,
            )
            .unwrap(),
            "case {case}, rows {rows}"
        );
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
        assert_eq!(
            source_bytes,
            panel_rows * AMP_COLUMN_MICRO * precision.bytes() as u32
        );
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
            for work in tile.work.iter() {
                if let BlockOperation::Compute { run, .. } = work {
                    let run = &low.kernel_runs[run.0 as usize];
                    run.call().unwrap();
                    assert_eq!(
                        low.shards[run.outputs[0].shard.index() as usize].tile,
                        tile.tile
                    );
                    assert!(
                        run.inputs.iter().all(|view| {
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
            let add = tile
                .work
                .iter()
                .find_map(|work| match work {
                    BlockOperation::Compute { run, .. }
                        if matches!(
                            low.kernel_runs[run.0 as usize].kernel,
                            MidOperationKind::Add
                        ) =>
                    {
                        Some(&low.kernel_runs[run.0 as usize])
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(add.inputs[0].extents[0].logical_end, 1);
            assert!(
                !matches!(
                    low.shards[add.inputs[0].shard.index() as usize].definition,
                    ShardDefinition::Unmaterialized
                ),
                "broadcast must resolve locally reused input views"
            );
            assert_eq!(
                low.shards[add.inputs[0].shard.index() as usize].tile,
                tile.tile
            );
        }
    }
}

#[test]
fn randomized_blocked_gemms_expand_to_tile_kernel_phases() {
    let shared_cache = Arc::new(GeometryCache::default());
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
        let fresh = super::expand_tiles_cached(&mid, true, Arc::default()).unwrap();
        for _ in 0..2 {
            let cached = super::expand_tiles_cached(&mid, true, Arc::clone(&shared_cache)).unwrap();
            assert_eq!(cached, fresh, "cache changed graph in case {case}");
        }
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

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

        for tile in &low.tiles {
            let gemms = tile
                .work
                .iter()
                .filter_map(|work| match work {
                    BlockOperation::Compute { run, .. }
                        if matches!(
                            low.kernel_runs[run.0 as usize].kernel,
                            MidOperationKind::Gemm { .. }
                        ) =>
                    {
                        Some(&low.kernel_runs[run.0 as usize])
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            let mut initialized_columns = std::collections::BTreeSet::new();
            for run in gemms {
                assert_eq!(run.provenance.reason, WorkReason::OperatorKernel);
                assert!(run.provenance.operation.is_some());
                assert!(run.provenance.value.is_some());
                let MidOperationKind::Gemm {
                    mode,
                    inner_block: kernel_inner,
                    output_columns: kernel_columns,
                    ..
                } = run.kernel
                else {
                    unreachable!()
                };
                let output_key = run.outputs[0]
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
                assert!(
                    run.inputs[0]
                        .extents
                        .iter()
                        .any(|extent| { extent.physical_end - extent.start == kernel_inner })
                );
                assert!(
                    run.outputs[0]
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
        assert_eq!(
            low.value_views(low.outputs[0]).len(),
            usize::from(selected_tiles)
        );
        for view in low.value_views(low.outputs[0]) {
            let shard = view.shard;
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
            let Some(candidate) = candidate.concrete() else {
                return false;
            };
            matches!(
                candidate.plan.dispatch,
                OperatorDispatch::BlockedGemm {
                    distribution: GemmDistribution::OutputStationary,
                    ..
                }
            ) && candidate.plan.inputs.get(1).is_some_and(|requirement| {
                requirement.format.layout.order
                    == crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                        row_block: 64,
                        column_block: crate::tensor::AMP_COLUMN_MICRO as u16,
                    })
                    && requirement.format.layout.tiling.tile_count == tiles
                    && requirement.format.layout.memory_class == MemoryClass::Ipu21Interleaved
            })
        });
        let selected = crate::planner::build::select(
            &graph,
            &config,
            &Ipu21CostModel,
            &crate::planner::cache::FragmentCache::default(),
            false,
        )
        .unwrap();
        let product = selected
            .operations
            .iter()
            .find(|operation| matches!(operation.kind, MidOperationKind::Product(_)))
            .unwrap();
        let config = config
            .with_input(
                left,
                selected.values[product.inputs[0].index() as usize]
                    .tensor_type
                    .format
                    .clone(),
            )
            .with_input(
                right,
                selected.values[product.inputs[1].index() as usize]
                    .tensor_type
                    .format
                    .clone(),
            );
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
        assert!(low.tiles.iter().all(|tile| tile.work.iter().all(|work| {
            !matches!(work, BlockOperation::Copy { .. })
                && !matches!(work, BlockOperation::Exchange(_))
        })));
        assert!(
            low.tiles
                .iter()
                .flat_map(|tile| tile.work.iter())
                .any(|work| {
                    matches!(
                        work,
                        BlockOperation::Compute { run, .. }
                            if matches!(low.kernel_runs[run.0 as usize].kernel, MidOperationKind::Gemm {
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
            crate::planner::operator::LocalOperandStaging::Direct
        } else {
            crate::planner::operator::LocalOperandStaging::MatchRemote
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
                crate::tensor::GridOrder::ColumnsFast,
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
                crate::tensor::GridOrder::ColumnsFast,
            ),
        };
        let mut config = PipelineConfig::new(tiles)
            .with_input(left, left_format.clone())
            .with_input(right, right_format.clone());
        config.operator_candidates =
            vec![crate::planner::catalogue::ConcreteOperatorCandidate::new(
                crate::planner::operator::OperatorFamily::Gemm {
                    options: crate::GemmOptions::default(),
                    multiply: Precision::F16,
                    accumulate: crate::AccumulationPrecision::F32,
                },
                [
                    crate::planner::operator::OperandRequirement::new(left_format),
                    crate::planner::operator::OperandRequirement::new(right_format)
                        .with_local_staging(local_staging),
                ],
                output_format,
            )]
            .into_iter()
            .map(OperatorCandidate::Concrete)
            .collect();
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        let expected_weight_bytes = inner
            .div_ceil(u32::from(inner_partitions))
            .saturating_mul(columns.div_ceil(u32::from(column_partitions)))
            .saturating_mul(2);
        assert!(low.value_views(low.inputs[1].value).iter().all(|shard| {
            crate::shard_storage_bytes(&low.shards[shard.shard.index() as usize])
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
        // A sequence may contain more values than this invocation consumes.
        let parameters = (0..count + random.u32(0..=3))
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
            let repeats = tile
                .work
                .iter()
                .filter_map(|work| match work {
                    BlockOperation::Repeat(repeat) => Some(&low.repeat_runs[*repeat]),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(repeats.len(), 1, "case {case}");
            assert_eq!(repeats[0].count, count);
            assert_eq!(repeats[0].binding.iterated[0].inputs.len(), count as usize);
            let placement = crate::place(&low).unwrap();
            let stride = placement.sequence_strides[&repeats[0].binding.iterated[0].argument];
            assert!(stride > 0 && stride.is_multiple_of(4));
            let carried = &repeats[0].binding.carried[0];
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
                repeats[0]
                    .body
                    .work
                    .iter()
                    .any(|work| matches!(work, BlockOperation::Compute { .. }))
            );
        }
    }
}

#[test]
fn repeat_binds_every_linear_fragment_including_rotated_owners() {
    let mut graph = ComputeGraph::new();
    let carried = graph.host_input("carried", [8, 16]).unwrap();
    let invariant = graph.parameter("invariant", [8, 16]).unwrap();
    let parameters = (0..2)
        .map(|index| {
            graph
                .parameter(format!("parameter.{index}"), [8, 16])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let sequence = graph
        .value_sequence("parameters", parameters.clone())
        .unwrap();
    let result = graph
        .repeat(2, [carried], [invariant], [sequence], |body, arguments| {
            let first = body.add(arguments.carried[0], arguments.invariants[0])?;
            Ok(vec![body.add(first, arguments.iterated[0])?])
        })
        .unwrap()[0];
    graph.set_outputs([result]).unwrap();
    let format = TensorFormat {
        precision: Precision::F16,
        layout: Layout::logical_linear(2, 16),
    };
    let mut candidate = ConcreteOperatorCandidate::new(
        OperatorFamily::Add,
        vec![OperandRequirement::new(format.clone()); 2],
        format.clone(),
    );
    candidate.plan.reuse_inputs = Some(vec![0]);
    let mut config = PipelineConfig::new(2);
    for input in [carried, invariant].into_iter().chain(parameters) {
        config.inputs.insert(input, format.clone());
    }
    config.operator_candidates = vec![OperatorCandidate::Concrete(candidate)];
    let mut mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    for offset in 0..2 {
        for value in &mut mid.values {
            value.owners = crate::tensor::OwnerMap::rotated(offset);
        }
        let low = lower_to_tiles(&mid, false).unwrap();
        let placement = crate::place(&low).unwrap();
        for repeat in &low.repeat_runs {
            assert_eq!(repeat.binding.carried.len(), 4);
            assert_eq!(repeat.binding.invariants.len(), 4);
            assert_eq!(repeat.binding.iterated.len(), 4);
            for binding in &repeat.binding.carried {
                let initial = placement.shard_addresses[&binding.initial];
                for shard in [binding.argument, binding.yielded, binding.result] {
                    assert_eq!(placement.shard_addresses[&shard], initial);
                }
            }
            for binding in &repeat.binding.invariants {
                assert_eq!(
                    placement.shard_addresses[&binding.input],
                    placement.shard_addresses[&binding.argument]
                );
            }
            for binding in &repeat.binding.iterated {
                assert_eq!(
                    placement.shard_addresses[&binding.inputs[0]],
                    placement.shard_addresses[&binding.argument]
                );
                assert_eq!(binding.inputs.len(), 2);
                assert!(
                    placement.shard_addresses[&binding.inputs[1]]
                        > placement.shard_addresses[&binding.inputs[0]]
                );
            }
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
            let repeat = tile
                .work
                .iter()
                .find_map(|work| match work {
                    BlockOperation::Repeat(repeat) => Some(&low.repeat_runs[*repeat]),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                low.shards[repeat.binding.carried[0].yielded.index() as usize].definition,
                ShardDefinition::WritableAlias(repeat.binding.carried[0].argument),
                "case {case}"
            );
        }
    }
}

#[test]
fn repeat_copy_yield_reaches_the_carried_allocation() {
    use crate::{CoordinateMapping, MidInput, MidRegion, MidValue, OperandIndexing};
    let id = MidValueId::from_index;
    let tensor_type = TensorType::new([8, 16], Precision::F16, Layout::row_sharded(1));
    let operation = |inputs: &[u32], result, kind| MidOperation {
        source: None,
        inputs: inputs.iter().copied().map(id).collect(),
        results: vec![id(result)],
        operands: if matches!(kind, MidOperationKind::Gelu) {
            vec![OperandIndexing::Elementwise { result: 0 }]
        } else {
            vec![]
        },
        kind,
        output_aliases: Vec::new(),
    };
    let mid = MidProgram {
        tile_count: 1,
        values: (0..5)
            .map(|index| MidValue {
                id: id(index),
                owners: crate::tensor::OwnerMap::default(),
                origin: crate::ValueId::from_index(index),
                storage_group: id(index),
                tensor_type: tensor_type.clone(),
            })
            .collect(),
        inputs: vec![MidInput {
            name: "input".into(),
            kind: crate::GraphInputKind::Host,
            value: id(0),
        }],
        outputs: vec![id(4)],
        operations: vec![operation(
            &[0],
            4,
            MidOperationKind::Repeat(MidRepeat {
                count: 2,
                carried_inputs: 1,
                invariant_inputs: 0,
                iterated_inputs: vec![],
                body: MidRegion {
                    arguments: vec![id(1)],
                    yields: vec![id(3)],
                    operations: vec![
                        operation(&[1], 2, MidOperationKind::Gelu),
                        operation(
                            &[2],
                            3,
                            MidOperationKind::Copy {
                                policy: crate::CopyPolicy::Automatic,
                                packing: crate::PackingPolicy::Automatic,
                                mapping: CoordinateMapping::default(),
                                reuse_local: true,
                            },
                        ),
                    ],
                },
            }),
        )],
        ..MidProgram::default()
    };
    let low = lower_to_tiles(&mid, false).unwrap();
    let placement = crate::place(&low).unwrap();
    let repeat = &low.repeat_runs[0];
    let target = placement.shard_addresses[&repeat.binding.carried[0].result];
    assert!(
        repeat.body.work.iter().any(|work| match work {
            BlockOperation::Compute { run, .. } =>
                placement.shard_addresses[&low.kernel_runs[run.0 as usize].outputs[0].shard]
                    == target,
            BlockOperation::Copy { copy, .. } =>
                placement.shard_addresses[&low.local_copies[copy.0 as usize].movement().destination]
                    == target,
            _ => false,
        }),
        "the body must write the allocation carried into its next iteration"
    );
}

#[test]
fn repeat_preserves_shared_initial_values() {
    for case in 0..6 {
        let mut graph = ComputeGraph::new();
        let input = if case == 5 {
            graph.parameter("input", [8, 16]).unwrap()
        } else {
            graph.host_input("input", [8, 16]).unwrap()
        };
        let carried = if case == 2 {
            vec![input; 2]
        } else {
            vec![input]
        };
        let invariants = if case == 3 { vec![input] } else { vec![] };
        let results = graph
            .repeat(2, carried, invariants, [], |body, args| {
                args.carried
                    .iter()
                    .map(|&x| match args.invariants.first() {
                        Some(&bias) => body.add(x, bias),
                        None => body.gelu(x),
                    })
                    .collect()
            })
            .unwrap();
        let mut outputs = results.clone();
        if case == 0 {
            outputs.push(input);
        } else if case == 1 {
            outputs.push(graph.add(input, results[0]).unwrap());
        }
        graph.set_outputs(outputs).unwrap();
        let config = PipelineConfig::new(1).with_input(input, format(1));
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, false).unwrap();
        let placement = crate::place(&low).unwrap();
        let initial = placement.shard_addresses[&low.value_views(low.inputs[0].value)[0].shard];
        let repeat = &low.repeat_runs[0];
        for binding in &repeat.binding.carried {
            let result = placement.shard_addresses[&binding.result];
            if case == 4 {
                assert_eq!(initial, result, "an unshared host input can be donated");
            } else {
                assert_ne!(
                    initial, result,
                    "case {case}: Repeat overwrites shared input"
                );
            }
        }
        if case == 2 {
            assert_ne!(
                placement.shard_addresses[&repeat.binding.carried[0].result],
                placement.shard_addresses[&repeat.binding.carried[1].result],
                "two carried states must evolve independently"
            );
        }
    }
}

#[test]
fn repeat_rejects_overwriting_an_indirectly_live_carried_input() {
    let mut graph = ComputeGraph::new();
    let a = graph.host_input("a", [8, 16]).unwrap();
    let b = graph.host_input("b", [8, 16]).unwrap();
    let output = graph
        .repeat(2, [a, b], [], [], |body, args| {
            let t = body.gelu(args.carried[0])?;
            let y = body.gelu(args.carried[1])?;
            let z = body.gelu(t)?;
            Ok(vec![y, z])
        })
        .unwrap();
    graph.set_outputs(output).unwrap();
    let tensor_format = format(1);
    let mut candidate = ConcreteOperatorCandidate::new(
        OperatorFamily::Gelu,
        [OperandRequirement::new(tensor_format.clone())],
        tensor_format.clone(),
    );
    candidate.plan.reuse_inputs = Some(vec![0]);
    let mut config = PipelineConfig::new(1)
        .with_input(a, tensor_format.clone())
        .with_input(b, tensor_format);
    config.operator_candidates = vec![OperatorCandidate::Concrete(candidate)];
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    // y cannot overwrite a: t still lives there and is read by z afterwards.
    // Such cross-coupled state updates need a separate copy-back plan, which
    // this in-place Repeat representation does not support.
    assert_eq!(
        lower_to_tiles(&mid, false),
        Err(ExpansionError::RepeatRequiresInPlace(0))
    );
}

fn contains_phase(program: &LowProgram, list: &TileWorkList, phase: ExchangePhaseId) -> bool {
    list.work.iter().any(|work| match work {
        BlockOperation::Exchange(candidate) => *candidate == phase,
        BlockOperation::Repeat(repeat) => {
            contains_phase(program, &program.repeat_runs[*repeat].body, phase)
        }
        BlockOperation::Compute { .. }
        | BlockOperation::Copy { .. }
        | BlockOperation::Checkpoint(..) => false,
    })
}

#[test]
fn factor_copies_and_offset_windows_preserve_coordinates() {
    for chain in 0..5 {
        for offset in [0, 1] {
            for rank in 2..=5 {
                for split in 0..rank {
                    for merge in 0..rank {
                        if split == merge {
                            continue;
                        }
                        let mut shape = vec![2; rank];
                        shape[split] = 12;
                        let mut graph = ComputeGraph::new();
                        let input = graph.host_input("input", shape.clone()).unwrap();
                        let mut views = vec![AxisFactorView::new(split, merge, 3)];
                        match chain {
                            1 => views.push(AxisFactorView::new(split, merge, 2)),
                            2 => views.push(AxisFactorView::new(merge, split, 2)),
                            3 => views.push(views[0].inverse()),
                            4 => {
                                // Exercise an inverse without its matching producer.
                                // Both axes are divisible by three in this case.
                                views = vec![AxisFactorView::new(merge, split, 3).inverse()];
                            }
                            _ => {}
                        }
                        let mut output = input;
                        for &view in &views {
                            output = graph.view(output, view).unwrap();
                        }
                        graph.set_outputs([output]).unwrap();
                        let config = PipelineConfig::new(1).with_input(
                            input,
                            TensorFormat {
                                precision: Precision::F32,
                                layout: Layout::row_major(TensorTiling::replicated(1)),
                            },
                        );
                        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
                        let mut mid = mid;
                        assert_eq!(
                            mid.operations.len(),
                            if chain == 2 || chain == 3 { 2 } else { 1 }
                        );
                        let result = mid.outputs[0];
                        for width in &mut mid.values[result.index() as usize].tensor_type.shape.0 {
                            *width -= offset;
                        }
                        let copy = mid
                            .operations
                            .iter_mut()
                            .find(|operation| operation.results.contains(&result))
                            .unwrap();
                        let MidOperationKind::Copy { mapping, .. } = &mut copy.kind else {
                            panic!("view must resolve to a copy");
                        };
                        mapping.offsets = vec![offset; rank];

                        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();
                        let mut buffers = low
                            .shards
                            .iter()
                            .map(|shard| {
                                vec![
                                    u32::MAX;
                                    crate::shard_storage_bytes(shard).unwrap() as usize / 4
                                ]
                            })
                            .collect::<Vec<_>>();
                        let input_shard =
                            low.value_views(low.inputs[0].value)[0].shard.index() as usize;
                        buffers[input_shard] = (0..shape.iter().product::<u32>()).collect();
                        for work in low.tiles[0].work.iter() {
                            let copy = match work {
                                BlockOperation::Copy { copy, .. } => {
                                    low.local_copies[copy.0 as usize].movement()
                                }
                                BlockOperation::Exchange(phase) => {
                                    assert!(
                                        low.exchange_phases[phase.index() as usize]
                                            .transfers
                                            .is_empty()
                                    );
                                    continue;
                                }
                                _ => panic!("row-major view should only require copies"),
                            };
                            let (rows, width, source_stride, destination_stride) =
                                match copy.pattern {
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
                                    let value = buffers[copy.source.index() as usize][((copy
                                        .source_offset
                                        + row * source_stride
                                        + byte)
                                        / 4)
                                        as usize];
                                    buffers[copy.destination.index() as usize][((copy
                                        .destination_offset
                                        + row * destination_stride
                                        + byte)
                                        / 4)
                                        as usize] = value;
                                }
                            }
                        }
                        let window_shape = &mid.values[result.index() as usize].tensor_type.shape.0;
                        let actual =
                            &buffers[low.value_views(low.outputs[0])[0].shard.index() as usize];
                        for source in 0..shape.iter().product::<u32>() {
                            let mut index = source;
                            let mut coordinates = vec![0; rank];
                            for axis in (0..rank).rev() {
                                coordinates[axis] = index % shape[axis];
                                index /= shape[axis];
                            }
                            let mut current_shape = crate::TensorShape(shape.clone());
                            for &view in &views {
                                if view.reversed {
                                    let part = coordinates[view.merge_axis] % view.factor;
                                    coordinates[view.merge_axis] /= view.factor;
                                    coordinates[view.split_axis] +=
                                        part * current_shape.0[view.split_axis];
                                    current_shape = view.output_shape(&current_shape).unwrap();
                                } else {
                                    current_shape = view.output_shape(&current_shape).unwrap();
                                    let width = current_shape.0[view.split_axis];
                                    let part = coordinates[view.split_axis] / width;
                                    coordinates[view.split_axis] %= width;
                                    coordinates[view.merge_axis] =
                                        coordinates[view.merge_axis] * view.factor + part;
                                }
                            }
                            if coordinates.iter().any(|&coordinate| coordinate < offset) {
                                continue;
                            }
                            for coordinate in &mut coordinates {
                                *coordinate -= offset;
                            }
                            let target = coordinates
                                .iter()
                                .zip(window_shape)
                                .fold(0, |index, (&coordinate, &width)| index * width + coordinate);
                            assert_eq!(
                                actual[target as usize], source,
                                "rank {rank}, split {split}, merge {merge}, offset {offset}"
                            );
                        }
                    }
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

#[test]
fn in_place_pointwise_handles_multiple_linear_shards_per_tile() {
    let format = TensorFormat {
        precision: Precision::F16,
        layout: Layout::amp_left(16, 4).with_retained_order_linear_ownership(4, 16),
    };
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [3, 17, 32]).unwrap();
    let output = graph.gelu(input).unwrap();
    graph.set_outputs([output]).unwrap();
    let mut candidate = ConcreteOperatorCandidate::new(
        OperatorFamily::Gelu,
        [OperandRequirement::new(format.clone())],
        format.clone(),
    );
    candidate.plan.reuse_inputs = Some(vec![0]);
    let mut config = PipelineConfig::new(4).with_input(input, format);
    config.operator_candidates = vec![candidate]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&mid, false).unwrap();
    assert!(low.kernel_runs.len() > 4);
    for run in &low.kernel_runs {
        let result = &low.shards[run.outputs[0].shard.index() as usize];
        let ShardDefinition::WritableAlias(source) = result.definition else {
            panic!("expected in-place GeLU: {result:?}");
        };
        assert_eq!(low.shards[source.index() as usize].extents, result.extents);
    }
}

#[test]
fn local_casts_pair_corresponding_linear_fragments() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [8, 16]).unwrap();
    let output = graph.gelu(input).unwrap();
    graph.set_outputs([output]).unwrap();
    let target = TensorFormat {
        precision: Precision::F16,
        layout: Layout::logical_linear(2, 16),
    };
    let candidate = ConcreteOperatorCandidate::new(
        OperatorFamily::Gelu,
        [OperandRequirement::new(target.clone())],
        target.clone(),
    );
    let mut config = PipelineConfig::new(2).with_input(
        input,
        TensorFormat {
            precision: Precision::F32,
            ..target
        },
    );
    config.operator_candidates = vec![OperatorCandidate::Concrete(candidate)];
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&mid, false).unwrap();
    let casts = low
        .kernel_runs
        .iter()
        .filter(|run| {
            matches!(
                run.kernel,
                MidOperationKind::Cast {
                    from: Precision::F32,
                    to: Precision::F16
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(casts.len(), 8);
    for run in casts {
        assert_eq!(run.inputs[0].extents, run.outputs[0].extents);
    }
}

#[test]
fn complete_panel_grid_stays_one_logical_exchange() {
    let mut state = TileGraphBuilder::new(
        &MidProgram {
            tile_count: 2,
            ..MidProgram::default()
        },
        Arc::default(),
    )
    .unwrap();
    for (tile, columns) in [(0, 128), (1, 64)] {
        let tensor_type = TensorType::new(
            [64, columns],
            Precision::F16,
            Layout::amp_left(columns as u16, 1),
        );
        state
            .push_shard(BlockValue {
                id: BlockValueId(0),
                tile,
                tensor_type,
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: 64,
                        physical_end: 64,
                    },
                    ShardExtent {
                        axis: 1,
                        start: 0,
                        logical_end: columns,
                        physical_end: columns,
                    },
                ],
                definition: ShardDefinition::Staging,
            })
            .unwrap();
    }
    let source = state
        .narrow_view(&state.full_view(BlockValueId(0)), &[(1, 0, 64)])
        .unwrap();
    let destination = state.full_view(BlockValueId(1));
    let (mappings, order) = state
        .micro_panel_mappings(vec![(source.clone(), destination.clone())])
        .unwrap()
        .unwrap();
    assert_eq!(order, CopyOrder::Panels);
    assert_eq!(mappings.len(), 1);
    let provenance = WorkProvenance {
        operation: None,
        value: None,
        reason: WorkReason::LayoutRearrangement,
    };
    let mut batch = movement::MaterializationBatch::default();
    let mut body = BlockRegion::default();
    state
        .prepare_mapped_views(
            mappings,
            order,
            order,
            PackingPolicy::Automatic,
            provenance,
            &mut batch,
            &mut body,
        )
        .unwrap();
    state
        .append_materialization(batch, provenance, &mut body)
        .unwrap();
    assert_eq!(state.phases.len(), 1);
    let [transfer] = state.phases[0].transfers.as_slice() else {
        panic!("expanded panel grid");
    };
    let pairs = |order| {
        view_byte_traversal(&state.shards[0], &source, order)
            .unwrap()
            .spans()
            .flat_map(|s| s.offset..s.offset + s.bytes)
            .zip(
                view_byte_traversal(&state.shards[1], &destination, order)
                    .unwrap()
                    .spans()
                    .flat_map(|s| s.offset..s.offset + s.bytes),
            )
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        pairs(transfer.span_order(&state.shards)),
        pairs(CopyOrder::Semantic)
    );
}

#[test]
fn fp8_clipped_panels_do_not_fragment_regular_destinations() {
    let mut state = TileGraphBuilder::new(
        &MidProgram {
            tile_count: 3,
            ..MidProgram::default()
        },
        Arc::default(),
    )
    .unwrap();
    for tile in 0..3 {
        let mut layout = Layout::row_sharded(1);
        layout.order = if tile == 0 {
            ElementOrder::Amp(crate::AmpOrder::TransposedLeft)
        } else {
            ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            })
        };
        state
            .push_shard(BlockValue {
                id: BlockValueId(0),
                tile,
                tensor_type: TensorType::new(
                    [64, 16],
                    Precision::F8F143 { scale_exponent: -4 },
                    layout,
                ),
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: 64,
                        physical_end: 64,
                    },
                    ShardExtent {
                        axis: 1,
                        start: 0,
                        logical_end: 16,
                        physical_end: 16,
                    },
                ],
                definition: ShardDefinition::Staging,
            })
            .unwrap();
    }
    let source = state.full_view(BlockValueId(0));
    let target = state.full_view(BlockValueId(1));
    let clipped_source = state
        .narrow_view(&state.full_view(BlockValueId(0)), &[(1, 0, 12)])
        .unwrap();
    let clipped_target = state
        .narrow_view(&state.full_view(BlockValueId(2)), &[(1, 0, 12)])
        .unwrap();
    let parts = super::mapping::split_mapping_at_panel_boundaries(
        &state.shards[0],
        clipped_source.clone(),
        &state.shards[2],
        clipped_target.clone(),
    )
    .unwrap();
    assert_eq!(parts.len(), 2, "32x12 FP8 panels, not four 16x12 halves");
    let mut actual = BTreeSet::new();
    for (a, b) in &parts {
        let a = view_byte_traversal(&state.shards[0], a, CopyOrder::Physical).unwrap();
        let b = view_byte_traversal(&state.shards[2], b, CopyOrder::Physical).unwrap();
        assert_eq!(a.spans().count(), 1);
        actual.extend(
            a.spans()
                .flat_map(|s| s.offset..s.offset + s.bytes)
                .zip(b.spans().flat_map(|s| s.offset..s.offset + s.bytes)),
        );
    }
    let a = view_byte_traversal(&state.shards[0], &clipped_source, CopyOrder::Semantic).unwrap();
    let b = view_byte_traversal(&state.shards[2], &clipped_target, CopyOrder::Semantic).unwrap();
    let expected = a
        .spans()
        .flat_map(|s| s.offset..s.offset + s.bytes)
        .zip(b.spans().flat_map(|s| s.offset..s.offset + s.bytes))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    let provenance = WorkProvenance {
        operation: None,
        value: None,
        reason: WorkReason::LayoutRearrangement,
    };
    let mut batch = movement::MaterializationBatch::default();
    let mut body = BlockRegion::default();
    state
        .prepare_mapped_views(
            vec![(source, target), (clipped_source, clipped_target)],
            CopyOrder::Semantic,
            CopyOrder::Semantic,
            PackingPolicy::Automatic,
            provenance,
            &mut batch,
            &mut body,
        )
        .unwrap();
    state
        .append_materialization(batch, provenance, &mut body)
        .unwrap();
    let regular = state
        .phases
        .iter()
        .flat_map(|p| &p.transfers)
        .filter(|t| t.destinations.iter().any(|d| d.shard == BlockValueId(1)))
        .collect::<Vec<_>>();
    assert_eq!(regular.len(), 1);
    assert_eq!(regular[0].order, CopyOrder::Panels);
}

#[test]
fn copy_preparation_preserves_order_when_distinct_values_share_storage() {
    fn mapping(copies: &[(u16, LocalCopy)]) -> Vec<(u32, u32)> {
        copies
            .iter()
            .flat_map(|(_, copy)| {
                let (rows, bytes, source_stride, destination_stride) = match copy.pattern {
                    crate::CopyPattern::Contiguous => (1, copy.bytes, 0, 0),
                    crate::CopyPattern::Strided {
                        rows,
                        row_bytes,
                        source_stride,
                        destination_stride,
                    } => (rows, row_bytes, source_stride, destination_stride),
                };
                (0..rows).flat_map(move |row| {
                    (0..bytes).map(move |byte| {
                        (
                            copy.source_offset + row * source_stride + byte,
                            copy.destination_offset + row * destination_stride + byte,
                        )
                    })
                })
            })
            .collect()
    }
    let orders = [
        ElementOrder::RowMajor,
        ElementOrder::Amp(crate::AmpOrder::Left),
        ElementOrder::Amp(crate::AmpOrder::Output),
        ElementOrder::Amp(crate::AmpOrder::TransposedLeft),
        ElementOrder::Amp(crate::AmpOrder::TransposedOutput),
        ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
            row_block: 16,
            column_block: 32,
        }),
    ];
    let mut reordered = 0;
    for precision in [Precision::F16, Precision::F32] {
        for source_order in orders {
            for destination_order in orders {
                let mut shards = [source_order, destination_order]
                    .into_iter()
                    .enumerate()
                    .map(|(index, order)| {
                        let mut layout = Layout::row_sharded(1);
                        layout.order = order;
                        BlockValue {
                            id: BlockValueId(index as u32),
                            tile: 0,
                            tensor_type: TensorType::new([48, 96], precision, layout),
                            extents: [48, 96]
                                .into_iter()
                                .enumerate()
                                .map(|(axis, end)| ShardExtent {
                                    axis: axis as u16,
                                    start: 0,
                                    logical_end: end,
                                    physical_end: end,
                                })
                                .collect(),
                            definition: ShardDefinition::Staging,
                        }
                    })
                    .collect::<Vec<_>>();
                let view = |id| ShardView {
                    shard: BlockValueId(id),
                    extents: vec![
                        ShardExtent {
                            axis: 0,
                            start: 0,
                            logical_end: 32,
                            physical_end: 32,
                        },
                        ShardExtent {
                            axis: 1,
                            start: 0,
                            logical_end: 64,
                            physical_end: 64,
                        },
                    ],
                };
                let (source, destination) = (view(0), view(1));
                let a = source
                    .bind(&shards)
                    .unwrap()
                    .traversal(CopyOrder::Semantic)
                    .unwrap();
                let b = destination
                    .bind(&shards)
                    .unwrap()
                    .traversal(CopyOrder::Semantic)
                    .unwrap();
                let mut expected = Vec::new();
                crate::storage::for_each_copy_span(a.spans(), b.spans(), |a, b, bytes| {
                    expected.extend((0..bytes).map(|byte| (a + byte, b + byte)));
                    Ok(())
                })
                .unwrap();
                let cache = GeometryCache::default();
                let mut separate = Vec::new();
                append_span_copies(
                    &cache,
                    &shards,
                    &source,
                    &destination,
                    0,
                    &mut separate,
                    CopyOrder::Semantic,
                )
                .unwrap();
                reordered += usize::from(mapping(&separate) != expected);
                for offset in [0, 64] {
                    shards[1].definition = ShardDefinition::ShiftedAlias {
                        source: BlockValueId(0),
                        offset,
                    };
                    let mut shared = Vec::new();
                    append_span_copies(
                        &cache,
                        &shards,
                        &source,
                        &destination,
                        0,
                        &mut shared,
                        CopyOrder::Semantic,
                    )
                    .unwrap();
                    assert_eq!(mapping(&shared), expected);
                }
            }
        }
    }
    assert!(
        reordered > 0,
        "fixture must expose unsafe reuse of a reordered disjoint copy"
    );
}
