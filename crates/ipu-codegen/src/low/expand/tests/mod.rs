use super::copy::{
    mapping::*,
    realize::{MaterializationBatch, append_span_copies},
};
use crate::mid::MidOperationKind;
use crate::tensor::{AMP_INNER_BLOCK, BlockMajorOrder};
use ipu_target::Target;
fn lower_to_tiles(
    graph: &crate::MidGraph,
    checkpoints: bool,
) -> super::ExpansionResult<crate::LowGraph> {
    let mut graph = graph.clone();
    graph.compose_copies();
    let expanded = super::expand_tiles_cached(Target::Ipu21, &graph, true, false, Arc::default())?;
    Ok(crate::low::lower_to_tiles(&expanded, checkpoints))
}
use super::*;
use crate::{
    AxisTiling, ElementOrder, GridOrder, HighGraph, Layout, MemoryClass, Padding, PipelineConfig,
    Precision, TensorAxis, TensorFormat, TensorTiling,
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
            let mut graph = HighGraph::new();
            let input = graph.host_input("input", [16, 16]).unwrap();
            let output = graph.gelu(input).unwrap();
            graph.set_outputs([output]).unwrap();
            let config = PipelineConfig::new(Target::Ipu21, 4).with_input(input, format(1));
            let mid = crate::planner::plan(
                &graph,
                &crate::planner::boundary_layouts(&graph, &config),
                &config,
                crate::planner::SearchLimits::default(),
            )
            .unwrap();
            let mut builder = TileGraphBuilder::new(Target::Ipu21, &mid, Arc::default()).unwrap();
            let mut ids = vec![builder.program.shards[0].id];
            for tile in [1, 2, 1] {
                let mut shard = builder.program.shards[0].clone();
                shard.tile = tile;
                shard.definition = ShardDefinition::Staging;
                ids.push(builder.push_shard(shard).unwrap());
            }
            let mut alias_shard = builder.program.shards[ids[1].index() as usize].clone();
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
            builder.program.local_copies.push(
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
                    &builder.program.shards,
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
            assert_eq!(builder.program.exchange_phases.len(), 2);
            crate::low::passes::movement::group_exchanges(
                &mut region,
                &mut builder.program.exchange_phases,
                &builder.program.local_copies,
                &builder.program.shards,
            )
            .unwrap();
            assert_eq!(
                crate::low::passes::movement::group_exchanges(
                    &mut region,
                    &mut builder.program.exchange_phases,
                    &builder.program.local_copies,
                    &builder.program.shards
                )
                .unwrap(),
                0
            );
            assert_eq!(
                builder.program.exchange_phases.len(),
                if blocked { 2 } else { 1 },
                "alias={alias}, source={source}, destination={destination}, offset={offset}"
            );
            assert_eq!(
                matches!(region.operations[0], BlockOperation::Copy { .. }),
                hoisted
            );
            if !blocked {
                assert_eq!(builder.program.exchange_phases[0].transfers.len(), 2);
                assert_eq!(
                    builder.program.exchange_phases[0].transfers[0].source.shard,
                    ids[0]
                );
                assert_eq!(
                    builder.program.exchange_phases[0].transfers[1].source.shard,
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
        let mut graph = HighGraph::new();
        let input = graph.host_input("input", [16, 16]).unwrap();
        graph.set_outputs([input]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, 3).with_input(input, format(1));
        let mid = crate::planner::plan(
            &graph,
            &crate::planner::boundary_layouts(&graph, &config),
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
        let mut builder = TileGraphBuilder::new(Target::Ipu21, &mid, Arc::default()).unwrap();
        let source = builder.full_view(builder.program.shards[0].id);
        let mut mappings = Vec::new();
        for tile in 0..=remote_count {
            let mut destination = builder.program.shards[0].clone();
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
        let mut batch = MaterializationBatch::default();
        let mut region = BlockRegion::default();
        builder
            .prepare_mapped_views(
                mappings,
                CopyOrder::Physical,
                exchange_order,
                PackingPolicy::Staged,
                provenance,
                &mut batch,
                &mut region,
            )
            .unwrap();
        builder
            .append_materialization(batch, provenance, &mut region, false)
            .unwrap();
        assert_eq!(builder.program.local_copies.is_empty(), loopback);
        let self_receivers = builder
            .program
            .exchange_phases
            .iter()
            .flat_map(|phase| &phase.transfers)
            .flat_map(|transfer| &transfer.destinations)
            .filter(|view| builder.program.shards[view.shard.index() as usize].tile == 0)
            .count();
        assert_eq!(self_receivers, usize::from(loopback));
        if loopback {
            builder.program.body = region;
            let graph = builder.program;
            let low = crate::low::lower_to_tiles(&Arc::new(graph), false);
            // Force standard storage into the shared upper region as well.
            let placement = crate::place::place_with_ranges(
                Target::Ipu21,
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
    let mut graph = HighGraph::new();
    let input = graph.host_input("input", [1, 4, 32]).unwrap();
    graph.set_outputs([input]).unwrap();
    let config = PipelineConfig::new(Target::Ipu21, 1).with_input(input, format(1));
    let mid = crate::planner::plan(
        &graph,
        &crate::planner::boundary_layouts(&graph, &config),
        &config,
        crate::planner::SearchLimits::default(),
    )
    .unwrap();
    let mut builder = TileGraphBuilder::new(Target::Ipu21, &mid, Arc::default()).unwrap();
    let source = builder.program.shards[0].id;
    let mut source_view = builder.full_view(source);
    source_view.extents = crate::OperandWindow(vec![(1, 0, 2)])
        .select(&source_view.extents, false)
        .unwrap();
    let source_shape = crate::TensorShape(vec![1, 2, 32]);
    let view = crate::AxisFactorView::new(2, 0, 2);
    let mut destination = builder.program.shards[0].clone();
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
    let mut batch = MaterializationBatch::default();
    builder
        .prepare_mapped_views(
            mappings,
            CopyOrder::Semantic,
            CopyOrder::Semantic,
            PackingPolicy::Staged,
            provenance,
            &mut batch,
            &mut region,
        )
        .unwrap();
    builder
        .append_materialization(batch, provenance, &mut region, false)
        .unwrap();
    assert!(!builder.program.local_copies.is_empty());
    assert!(
        builder
            .program
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
}

fn format(tiles: u16) -> TensorFormat {
    TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(tiles),
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
fn randomized_pointwise_dispatch_skips_empty_output_shards() {
    let mut random = fastrand::Rng::with_seed(0x656d_7074);
    for case in 0..CASES {
        let tiles = random.u16(2..=32);
        let rows = random.u32(1..u32::from(tiles));
        let columns = random.u32(1..=32) * 2;
        let tensor_format = format(tiles);
        let mut graph = HighGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, tiles).with_input(input, tensor_format);

        let mid = crate::planner::plan(
            &graph,
            &crate::planner::boundary_layouts(&graph, &config),
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
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
        let mut graph = HighGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, tiles).with_input(input, input_format);
        let mut layouts = crate::planner::boundary_layouts(&graph, &config);
        layouts.insert(output, Some(output_format.layout));
        let mid = crate::planner::plan(
            &graph,
            &layouts,
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
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
    for _ in 0..CASES {
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
        let mut graph = HighGraph::new();
        let input = graph.host_input("input", [rows, columns]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, tiles).with_input(input, input_format);
        let mut layouts = crate::planner::boundary_layouts(&graph, &config);
        layouts.insert(output, Some(target_format.layout));
        let mid = crate::planner::plan(
            &graph,
            &layouts,
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
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
            definition: ShardDefinition::Staging,
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
            definition: ShardDefinition::Staging,
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
        let mut graph = HighGraph::new();
        let left = graph.host_input("left", [rows, columns]).unwrap();
        let right = graph.host_input("right", [rows, columns]).unwrap();
        let output = graph.add(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, tiles)
            .with_input(left, format(tiles))
            .with_input(right, format(tiles));
        let mid = crate::planner::plan(
            &graph,
            &crate::planner::boundary_layouts(&graph, &config),
            &config,
            crate::planner::SearchLimits::default(),
        )
        .unwrap();
        let low = lower_to_tiles(&mid, config.diagnostic_checkpoints).unwrap();

        assert_eq!(low.tiles.len(), usize::from(tiles), "case {case}");
        for tile in &low.tiles {
            for work in tile.work.iter() {
                if let BlockOperation::Compute { run, .. } = work {
                    let run = &low.kernel_runs[run.0 as usize];
                    run.call(Target::Ipu21, None).unwrap();
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
                    ShardDefinition::Value(_) | ShardDefinition::Staging
                )));
            }
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
        output_windows: Vec::new(),
    };
    let mid = MidGraph {
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
                                packing: crate::PackingPolicy::Staged,
                                mapping: CoordinateMapping::default(),
                            },
                        ),
                    ],
                },
            }),
        )],
        ..MidGraph::default()
    };
    let low = lower_to_tiles(&mid, false).unwrap();
    let placement = crate::place(Target::Ipu21, &low).unwrap();
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

fn contains_phase(program: &LowGraph, list: &TileWorkList, phase: ExchangePhaseId) -> bool {
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
    let mut graph = HighGraph::new();
    let input = graph.host_input("input", [3, 17, 32]).unwrap();
    let output = graph.gelu(input).unwrap();
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(Target::Ipu21, 4).with_input(input, format.clone());
    let mut layouts = crate::planner::boundary_layouts(&graph, &config);
    layouts.insert(output, Some(format.layout));
    let mut mid = crate::planner::plan(
        &graph,
        &layouts,
        &config,
        crate::planner::SearchLimits::default(),
    )
    .unwrap();
    mid.operations[0].output_aliases = vec![(0, 0)];
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
fn complete_panel_grid_stays_one_logical_exchange() {
    let mut state = TileGraphBuilder::new(
        Target::Ipu21,
        &MidGraph {
            tile_count: 2,
            ..MidGraph::default()
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
    let mut source = state.full_view(BlockValueId(0));
    source.extents = crate::OperandWindow(vec![(1, 0, 64)])
        .select(&source.extents, false)
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
    let mut batch = MaterializationBatch::default();
    let mut body = BlockRegion::default();
    state
        .prepare_mapped_views(
            mappings,
            order,
            order,
            PackingPolicy::Staged,
            provenance,
            &mut batch,
            &mut body,
        )
        .unwrap();
    state
        .append_materialization(batch, provenance, &mut body, false)
        .unwrap();
    assert_eq!(state.program.exchange_phases.len(), 1);
    let [transfer] = state.program.exchange_phases[0].transfers.as_slice() else {
        panic!("expanded panel grid");
    };
    let pairs = |order| {
        view_byte_traversal(&state.program.shards[0], &source, order)
            .unwrap()
            .spans()
            .flat_map(|s| s.offset..s.offset + s.bytes)
            .zip(
                view_byte_traversal(&state.program.shards[1], &destination, order)
                    .unwrap()
                    .spans()
                    .flat_map(|s| s.offset..s.offset + s.bytes),
            )
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        pairs(transfer.span_order(&state.program.shards)),
        pairs(CopyOrder::Semantic)
    );
}

#[test]
fn fp8_clipped_panels_do_not_fragment_regular_destinations() {
    let mut state = TileGraphBuilder::new(
        Target::Ipu21,
        &MidGraph {
            tile_count: 3,
            ..MidGraph::default()
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
    let mut clipped_source = state.full_view(BlockValueId(0));
    clipped_source.extents = crate::OperandWindow(vec![(1, 0, 12)])
        .select(&clipped_source.extents, false)
        .unwrap();
    let mut clipped_target = state.full_view(BlockValueId(2));
    clipped_target.extents = crate::OperandWindow(vec![(1, 0, 12)])
        .select(&clipped_target.extents, false)
        .unwrap();
    let parts = super::copy::mapping::split_mapping_at_panel_boundaries(
        &state.program.shards[0],
        clipped_source.clone(),
        &state.program.shards[2],
        clipped_target.clone(),
    )
    .unwrap();
    assert_eq!(parts.len(), 2, "32x12 FP8 panels, not four 16x12 halves");
    let mut actual = BTreeSet::new();
    for (a, b) in &parts {
        let a = view_byte_traversal(&state.program.shards[0], a, CopyOrder::Physical).unwrap();
        let b = view_byte_traversal(&state.program.shards[2], b, CopyOrder::Physical).unwrap();
        assert_eq!(a.spans().count(), 1);
        actual.extend(
            a.spans()
                .flat_map(|s| s.offset..s.offset + s.bytes)
                .zip(b.spans().flat_map(|s| s.offset..s.offset + s.bytes)),
        );
    }
    let a = view_byte_traversal(
        &state.program.shards[0],
        &clipped_source,
        CopyOrder::Semantic,
    )
    .unwrap();
    let b = view_byte_traversal(
        &state.program.shards[2],
        &clipped_target,
        CopyOrder::Semantic,
    )
    .unwrap();
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
    let mut batch = MaterializationBatch::default();
    let mut body = BlockRegion::default();
    state
        .prepare_mapped_views(
            vec![(source, target), (clipped_source, clipped_target)],
            CopyOrder::Semantic,
            CopyOrder::Semantic,
            PackingPolicy::Direct,
            provenance,
            &mut batch,
            &mut body,
        )
        .unwrap();
    state
        .append_materialization(batch, provenance, &mut body, false)
        .unwrap();
    let regular = state
        .program
        .exchange_phases
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
