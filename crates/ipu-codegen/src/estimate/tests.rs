use super::*;
use crate::{AMP_INNER_BLOCK, GemmDistribution, OperatorDispatch, TensorAxis};
use std::collections::BTreeSet;

fn output_stationary_dispatch() -> OperatorDispatch {
    OperatorDispatch::BlockedGemm {
        inner_block: AMP_INNER_BLOCK,
        output_column_block: crate::mid::AMP_OUTPUT_COLUMN_BLOCK,
        orientation: crate::GemmOrientation::Normal,
        distribution: GemmDistribution::OutputStationary,
    }
}

fn parallel_reduction_dispatch(
    row_partitions: u16,
    column_partitions: u16,
    inner_partitions: u16,
) -> OperatorDispatch {
    let mut dispatch = output_stationary_dispatch();
    let OperatorDispatch::BlockedGemm { distribution, .. } = &mut dispatch else {
        unreachable!();
    };
    *distribution = GemmDistribution::ParallelReduction {
        row_partitions,
        column_partitions,
        inner_partitions,
        result_row_partitions: 1,
        result_column_partitions: 1,
        reduction_staging: crate::ReductionStaging::Streamed,
    };
    dispatch
}

#[test]
fn randomized_average_shard_storage_covers_spatial_work() {
    let mut random = fastrand::Rng::with_seed(0x7370_6174_6961_6c77);
    for case in 0..32 {
        let row_partitions = 1_u16 << random.u32(0..=4);
        let column_partitions = 1_u16 << random.u32(0..=4);
        let tiles = row_partitions * column_partitions;
        let rows = u32::from(row_partitions) * random.u32(1..=8);
        let columns = u32::from(column_partitions) * 64 * random.u32(1..=4);
        let tensor = TensorType::new(
            [rows, columns],
            Precision::F16,
            Layout::amp_output_grid(
                64,
                tiles,
                row_partitions,
                column_partitions,
                crate::mid::GridOrder::ColumnsFast,
            ),
        );
        let total = physical_elements(&tensor.shape, &tensor.format.layout)
            .saturating_mul(tensor.format.precision.bytes());
        let average = total.div_ceil(u64::from(tiles));
        assert!(
            average.saturating_mul(u64::from(tiles)) >= total,
            "case {case}"
        );
        assert!(average <= maximum_shard_bytes(&tensor), "case {case}");
    }
}

fn conversion_traffic_reference(
    shape: &TensorShape,
    precision: Precision,
    from: &Layout,
    to: &Layout,
) -> ConversionTraffic {
    let sources = layout_extents(shape, from).unwrap();
    let destinations = layout_extents(shape, to).unwrap();
    let mut remote = BTreeSet::<(u16, Vec<(u32, u32)>)>::new();
    let mut traffic = ConversionTraffic::default();
    for (destination_tile, destination) in &destinations {
        let mut intersections = BTreeMap::<Vec<(u32, u32)>, u16>::new();
        for (source_tile, source) in &sources {
            let Some(extents) = intersect_ranges(source, destination) else {
                continue;
            };
            let selected = intersections.entry(extents).or_insert(*source_tile);
            if source_tile == destination_tile {
                *selected = *source_tile;
            }
        }
        let mut destination_bytes = 0;
        let mut local_bytes = 0;
        let mut local_intersections = 0;
        let mut remote_intersections = 0;
        for (extents, source_tile) in &intersections {
            let bytes = range_elements(extents) * precision.bytes();
            destination_bytes += bytes;
            if source_tile == destination_tile {
                local_bytes += bytes;
                local_intersections += 1;
            } else {
                remote.insert((*source_tile, extents.clone()));
                remote_intersections += 1;
            }
        }
        traffic.maximum_destination_bytes =
            traffic.maximum_destination_bytes.max(destination_bytes);
        traffic.maximum_remote_destination_bytes = traffic
            .maximum_remote_destination_bytes
            .max(destination_bytes.saturating_sub(local_bytes));
        traffic.maximum_remote_destination_fragments = traffic
            .maximum_remote_destination_fragments
            .max(remote_intersections);
        add_endpoint_load(
            &mut traffic.remote_destination_loads,
            *destination_tile,
            destination_bytes.saturating_sub(local_bytes),
            remote_intersections,
        );
        traffic.maximum_local_bytes = traffic.maximum_local_bytes.max(local_bytes);
        traffic.maximum_intersections = traffic
            .maximum_intersections
            .max(intersections.len() as u64);
        traffic.maximum_local_intersections =
            traffic.maximum_local_intersections.max(local_intersections);
    }
    traffic.remote_fragments = remote.len() as u64;
    traffic.maximum_routed_fragments = if from.order == to.order {
        traffic.maximum_intersections
    } else {
        traffic
            .maximum_destination_bytes
            .saturating_sub(traffic.maximum_local_bytes)
            .div_ceil(4)
    };
    traffic.source_payload_bytes = remote
        .iter()
        .map(|(_, extents)| range_elements(extents) * precision.bytes())
        .sum();
    let mut source_roles = BTreeMap::<u16, (u64, u64)>::new();
    for (source, extents) in &remote {
        let role = source_roles.entry(*source).or_default();
        role.0 += range_elements(extents) * precision.bytes();
        role.1 += 1;
    }
    for (bytes, fragments) in source_roles.into_values() {
        traffic.maximum_source_payload_bytes = traffic.maximum_source_payload_bytes.max(bytes);
        traffic.maximum_source_fragments = traffic.maximum_source_fragments.max(fragments);
    }
    let mut source_buses = BTreeMap::<u16, (u64, u64)>::new();
    for (source, extents) in &remote {
        let role = source_buses.entry(*source / 2).or_default();
        role.0 += range_elements(extents) * precision.bytes();
        role.1 += 1;
    }
    for (bus, (bytes, fragments)) in source_buses {
        traffic.maximum_source_bus_payload_bytes =
            traffic.maximum_source_bus_payload_bytes.max(bytes);
        traffic.maximum_source_bus_fragments = traffic.maximum_source_bus_fragments.max(fragments);
        add_endpoint_load(&mut traffic.source_bus_loads, bus, bytes, fragments);
    }
    traffic
}

#[test]
fn randomized_conversion_traffic_counts_fragmented_multicasts() {
    let mut random = fastrand::Rng::with_seed(0x6672_6167_6d65_6e74);
    for case in 0..32 {
        let row_partitions = 1_u16 << random.u32(1..=4);
        let column_partitions = 1_u16 << random.u32(1..=4);
        let tiles = row_partitions * column_partitions;
        let rows = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4);
        let columns = u32::from(row_partitions.max(column_partitions)) * random.u32(1..=4) * 64;
        let shape = TensorShape(vec![rows, columns]);
        let fragmented_source = Layout::amp_output_grid(
            64,
            tiles,
            row_partitions,
            column_partitions,
            crate::mid::GridOrder::ColumnsFast,
        );
        let aligned_source = Layout::amp_output_grid(
            64,
            tiles,
            column_partitions,
            row_partitions,
            crate::mid::GridOrder::ColumnsFast,
        );
        let destination =
            Layout::amp_output_replicated_grid(tiles, column_partitions, row_partitions);
        let fragmented =
            conversion_traffic(&shape, Precision::F16, &fragmented_source, &destination).unwrap();
        let aligned =
            conversion_traffic(&shape, Precision::F16, &aligned_source, &destination).unwrap();

        assert_eq!(
            fragmented,
            conversion_traffic_reference(&shape, Precision::F16, &fragmented_source, &destination,),
            "case {case}"
        );
        assert_eq!(
            aligned,
            conversion_traffic_reference(&shape, Precision::F16, &aligned_source, &destination,),
            "case {case}"
        );

        assert_eq!(
            fragmented.maximum_destination_bytes, aligned.maximum_destination_bytes,
            "case {case}"
        );
        assert!(
            fragmented.remote_fragments >= aligned.remote_fragments,
            "case {case}: {fragmented:?} {aligned:?}"
        );
        assert!(
            fragmented.maximum_intersections >= aligned.maximum_intersections,
            "case {case}: {fragmented:?} {aligned:?}"
        );
    }
}

#[test]
fn randomized_resolved_capacity_matches_physical_storage() {
    use crate::{AxisTiling, Padding, TensorTiling};
    let mut random = fastrand::Rng::with_seed(0x7265_736f_6c76_6564);
    for _ in 0..128 {
        let groups = random.u16(1..=4);
        let partitions = groups * random.u16(1..=4);
        let replicas = random.u16(1..=3);
        let rows = random.u32(1..=8);
        let columns = u32::from(groups) * random.u32(17..=96);
        let layout = Layout::row_major(TensorTiling {
            tile_count: partitions * replicas,
            replicas,
            axes: vec![
                AxisTiling::new(TensorAxis::FromEnd(1), partitions, 4, Padding::Zero)
                    .with_padding_groups(groups)
                    .with_shard_padding_multiple(16),
            ],
        });
        for layout in [layout, Layout::logical_linear(4, 1)] {
            let tensor = TensorType::new([rows, columns], Precision::F16, layout);
            let resolved = tensor.format.layout.resolve(&tensor.shape).unwrap();
            let mut tile_bytes = vec![0_u64; usize::from(tensor.format.layout.tiling.tile_count)];
            for (tile, extents) in resolved.shard_extents().unwrap() {
                let bytes = crate::shard_storage_bytes(&crate::BlockValue {
                    id: crate::BlockValueId::from_index(0),
                    tile,
                    tensor_type: tensor.clone(),
                    extents,
                    definition: crate::ShardDefinition::Staging,
                })
                .unwrap();
                tile_bytes[usize::from(tile)] += u64::from(bytes);
            }
            assert_eq!(
                resolved.maximum_tile_elements() * 2,
                *tile_bytes.iter().max().unwrap()
            );
            assert_eq!(
                resolved.physical_elements() * 2,
                tile_bytes.iter().sum::<u64>()
            );
        }
    }
}

#[test]
fn parallel_gemm_partial_capacity_uses_selected_ownership_grain() {
    use crate::{GemmOrientation, GridOrder};
    for orientation in [GemmOrientation::Normal, GemmOrientation::Swapped] {
        let (shape, layout) = match orientation {
            GemmOrientation::Normal => (
                [8, 160],
                Layout::amp_left_result_grid(16, 4, 1, 4, GridOrder::ColumnsFast),
            ),
            GemmOrientation::Swapped => (
                [160, 8],
                Layout::amp_transposed_left_result_grid(16, 4, 1, 4, GridOrder::ColumnsFast),
            ),
        };
        let mut output = TensorType::new(shape, Precision::F16, layout);
        let column_axis = orientation.matrix_axes(2).1;
        let axis = output
            .format
            .layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis.resolve(2) == Ok(column_axis))
            .unwrap();
        axis.shard_padding_multiple = 16;
        let mut dispatch = parallel_reduction_dispatch(1, 4, 2);
        let OperatorDispatch::BlockedGemm {
            orientation: selected,
            ..
        } = &mut dispatch
        else {
            unreachable!();
        };
        *selected = orientation;
        // The kernel handles up to 64 columns, but the four partials own
        // 48, 48, 32, and 32 columns. Kernel blocking must not repartition them.
        let partial = dispatch.gemm_partial_tensor(&output);
        assert_eq!(maximum_axis_shard_extent(&partial, column_axis), 48);
        assert_eq!(maximum_shard_bytes(&partial), 8 * 48 * 2);
        assert_eq!(
            physical_elements(&partial.shape, &partial.format.layout),
            8 * 160
        );
        assert!(
            !partial
                .format
                .layout
                .resolve(&partial.shape)
                .unwrap()
                .has_empty_shards()
        );
    }
}
