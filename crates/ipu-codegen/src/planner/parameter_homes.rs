//! Choose default persistent parameter homes before mid binds operand movement.

use crate::PipelineConfig;
use crate::mid::{MidValue, MidValueId};
use crate::planner::error::LoweringResult;
#[cfg(test)]
use crate::tensor::AmpOrder;
use crate::tensor::{ElementOrder, Layout, TensorType};
use ipu_target::ipu21::memory::IPU21_PLANNED_DATA_BYTES;
use std::collections::BTreeMap;

/// Keep small broadcasts in transfer-sized chunks rather than eight-byte
/// shards, unless the resident sequence would exceed the allocation budget.
pub(super) fn compact_parameter_layout(
    tensor: &TensorType,
    copies: u32,
    config: &PipelineConfig,
) -> Option<Layout> {
    let layout = if tensor.format.layout.order == ElementOrder::RowMajor {
        let grain = tensor.format.layout.tiling.linear_grain()?;
        let limit = config
            .tile_memory_budget_bytes
            .min(u64::from(IPU21_PLANNED_DATA_BYTES))
            .saturating_sub(config.standard_memory_reservation_bytes);
        let chunks_per_owner =
            limit / (u64::from(grain) * tensor.format.precision.bytes() * u64::from(copies));
        if chunks_per_owner == 0 {
            return None;
        }
        let required = tensor
            .shape
            .elements()
            .div_ceil(u64::from(grain))
            .div_ceil(chunks_per_owner);
        let owners = tensor
            .shape
            .elements()
            .saturating_mul(tensor.format.precision.bytes())
            .div_ceil(256)
            .min(u64::from(tensor.format.layout.tiling.tile_count))
            .max(required);
        if owners > u64::from(tensor.format.layout.tiling.tile_count) {
            return None;
        }
        Layout::logical_linear(owners as u16, grain)
    } else {
        compact_matrix_layout(tensor, config.tile_count)?
    };
    let bytes = layout
        .resolve(&tensor.shape)
        .ok()?
        .maximum_tile_elements()
        .saturating_mul(tensor.format.precision.bytes())
        .saturating_mul(u64::from(copies));
    (bytes.saturating_add(config.standard_memory_reservation_bytes)
        <= config
            .tile_memory_budget_bytes
            .min(u64::from(IPU21_PLANNED_DATA_BYTES)))
    .then_some(layout)
}

// Preserve micro-panel traversal, not the consumer's macro-panel dimensions.
// The existing panel exchange can regroup these fragments without permutation;
// retaining large macro panels needlessly restricts persistent ownership.
fn compact_matrix_layout(tensor: &TensorType, tiles: u16) -> Option<Layout> {
    use crate::{AmpOrder, AxisTiling, BlockMajorOrder, Padding, TensorAxis, TensorTiling};
    let dimensions = &tensor.shape.0;
    let (&rows, &columns) = (
        dimensions.get(dimensions.len().checked_sub(2)?)?,
        dimensions.last()?,
    );
    let micro = 32 / tensor.format.precision.bytes() as u32;
    let order = match tensor.format.layout.order {
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. }) => {
            ElementOrder::Amp(AmpOrder::TransposedLeft)
        }
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. }) => {
            ElementOrder::Amp(AmpOrder::Left)
        }
        order => order,
    };
    let (row_grain, column_grain) = match order {
        ElementOrder::RowMajor => return None,
        ElementOrder::Amp(AmpOrder::Left) => (16, micro),
        ElementOrder::Amp(AmpOrder::TransposedLeft) => (micro, 16),
        ElementOrder::Amp(AmpOrder::Output) => (1, 16),
        ElementOrder::Amp(AmpOrder::TransposedOutput) => (16, 1),
        ElementOrder::Amp(AmpOrder::TransposedRight) => (16, micro),
        ElementOrder::BlockMajor(_) => unreachable!(),
    };
    if row_grain == 0 || column_grain == 0 {
        return None;
    }
    let row_blocks = rows.div_ceil(row_grain);
    let column_blocks = columns.div_ceil(column_grain);
    let (_, _, row_parts, column_parts) = (1..=row_blocks.min(u32::from(tiles)))
        .filter_map(|row_parts| {
            let column_parts = column_blocks.min(u32::from(tiles) / row_parts);
            if column_parts == 0 {
                return None;
            }
            let shard = u64::from(row_blocks.div_ceil(row_parts))
                * u64::from(row_grain)
                * u64::from(column_blocks.div_ceil(column_parts))
                * u64::from(column_grain);
            Some((
                shard,
                shard * u64::from(row_parts * column_parts),
                row_parts,
                column_parts,
            ))
        })
        .min()?;
    Some(Layout {
        order,
        tiling: TensorTiling {
            tile_count: (row_parts * column_parts) as u16,
            replicas: 1,
            axes: vec![
                AxisTiling::new(
                    TensorAxis::FromEnd(2),
                    row_parts as u16,
                    row_grain,
                    Padding::Zero,
                ),
                AxisTiling::new(
                    TensorAxis::FromEnd(1),
                    column_parts as u16,
                    column_grain,
                    Padding::Zero,
                ),
            ],
        },
        memory_class: tensor.format.layout.memory_class,
    })
}

/// Select persistent homes before capacity screening. Sequence members share
/// one rotation; derived values follow that rotation but are not counted again.
pub(super) fn assign_parameter_tiles(
    values: &mut [MidValue],
    parameters: &[MidValueId],
    copies: &BTreeMap<MidValueId, u32>,
    tile_count: u16,
) -> LoweringResult<()> {
    let mut groups = BTreeMap::<MidValueId, Vec<u64>>::new();
    for &id in parameters {
        let value = &values[id.index() as usize];
        let layout = &value.tensor_type.format.layout;
        layout.validate_tile_count(tile_count)?;
        let resolved = layout.resolve(&value.tensor_type.shape)?;
        let bytes = groups
            .entry(value.storage_group)
            .or_insert_with(|| vec![0; usize::from(tile_count)]);
        for owner in 0..layout.tiling.tile_count {
            bytes[usize::from(owner)] = bytes[usize::from(owner)].saturating_add(
                resolved
                    .tile_elements(owner)
                    .saturating_mul(value.tensor_type.format.precision.bytes())
                    .saturating_mul(u64::from(copies.get(&id).copied().unwrap_or(1))),
            );
        }
    }
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(id, bytes)| {
        (
            std::cmp::Reverse(bytes.iter().copied().max().unwrap_or(0)),
            *id,
        )
    });
    let mut loads = vec![0u64; usize::from(tile_count)];
    let mut offsets = BTreeMap::new();
    for (group, bytes) in groups {
        let offset = balanced_offset(&loads, &bytes);
        for (owner, bytes) in bytes.into_iter().enumerate() {
            let load = &mut loads[(owner + usize::from(offset)) % usize::from(tile_count)];
            *load = load.saturating_add(bytes);
        }
        offsets.insert(group, offset);
    }
    for value in values.iter_mut() {
        if let Some(&offset) = offsets.get(&value.storage_group) {
            let owners = value.owners.with_rotation(offset);
            value.owners = owners;
        }
    }
    Ok(())
}

fn balanced_offset(loads: &[u64], bytes: &[u64]) -> u16 {
    if bytes.windows(2).all(|pair| pair[0] == pair[1]) && bytes.len() == loads.len() {
        return 0;
    }
    let peak = loads.iter().copied().max().unwrap_or(0);
    // Each shard affects a different tile. The old peak covers unchanged tiles,
    // so candidate scoring needs neither a cloned load array nor a full rescan.
    (0..loads.len())
        .step_by(loads.len().div_ceil(128))
        .min_by_key(|&offset| {
            let peak = bytes
                .iter()
                .enumerate()
                .fold(peak, |peak, (logical, &bytes)| {
                    peak.max(loads[(logical + offset) % loads.len()].saturating_add(bytes))
                });
            (peak, offset)
        })
        .unwrap_or(0) as u16
}

#[cfg(test)]
mod tests {
    use crate::graph::{GraphInputKind, ValueId};
    use crate::low::default_copy_policy;
    use crate::mid::{CoordinateMapping, MidInput, MidOperation, MidOperationKind, MidProgram};
    use crate::tensor::{MemoryClass, Precision, TensorShape};

    use super::*;

    #[test]
    fn compact_fp8_homes_preserve_panels_and_avoid_byte_permutations() {
        let config = PipelineConfig::new(64);
        let precision = Precision::F8F143 { scale_exponent: -4 };
        for order in [
            ElementOrder::Amp(AmpOrder::TransposedLeft),
            ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            }),
        ] {
            let mut target = Layout::amp_transposed_left_parallel_grid(64, 8, 2, 2, 2);
            target.order = order;
            let native = TensorType::new([128, 256], precision, target.clone());
            let home = compact_parameter_layout(&native, 27, &config).unwrap();
            assert_eq!(
                home.order.micro_panel_order(),
                target.order.micro_panel_order()
            );
            assert_eq!(home.tiling.replicas, 1);
            assert!(home.tiling.tile_count > target.tiling.tile_count);
            for (layout, encodable) in [(Layout::logical_linear(64, 8), false), (home, true)] {
                let source = TensorType::new([128, 256], precision, layout);
                let output = TensorType::new([128, 256], precision, target.clone());
                let program = MidProgram {
                    tile_count: 64,
                    inputs: vec![MidInput {
                        name: "weight".into(),
                        kind: GraphInputKind::Parameter,
                        value: MidValueId::from_index(0),
                    }],
                    values: [source.clone(), output.clone()]
                        .into_iter()
                        .enumerate()
                        .map(|(id, tensor_type)| MidValue {
                            id: MidValueId::from_index(id as u32),
                            owners: crate::tensor::OwnerMap::default(),
                            tensor_type,
                            origin: ValueId::from_index(0),
                            storage_group: MidValueId::from_index(id as u32),
                        })
                        .collect(),
                    operations: vec![MidOperation {
                        source: None,
                        inputs: vec![MidValueId::from_index(0)],
                        results: vec![MidValueId::from_index(1)],
                        kind: MidOperationKind::Copy {
                            mapping: CoordinateMapping::default(),
                            reuse_local: false,
                            packing: crate::PackingPolicy::Automatic,
                            policy: default_copy_policy(
                                &source.format.layout,
                                &output.format.layout,
                            ),
                        },
                    }],
                    outputs: vec![MidValueId::from_index(1)],
                    ..Default::default()
                };
                assert_eq!(crate::expand_tiles(&program).is_ok(), encodable);
            }
        }
    }

    #[test]
    fn compact_weight_homes_balance_whole_exchange_panels() {
        let config = PipelineConfig::new(1472);
        for shape in [[1152, 3456], [1152, 1152], [1152, 4304], [4304, 1152]] {
            let tensor = TensorType::new(
                shape,
                Precision::F8F143 { scale_exponent: -4 },
                Layout::block_major_matrix_storage(288, 16, 1, 1, 1, MemoryClass::Ipu21Standard),
            );
            let home = compact_parameter_layout(&tensor, 27, &config).unwrap();
            let shard = home.resolve(&tensor.shape).unwrap().maximum_tile_elements();
            let ideal = tensor
                .shape
                .elements()
                .div_ceil(u64::from(config.tile_count));
            assert!(
                shard * 100 <= ideal * 125,
                "{shape:?}: shard {shard}, ideal {ideal}"
            );
        }
    }

    #[test]
    fn parameter_homes_group_sequences_without_counting_temporary_derivatives() {
        let mut values = (0..5)
            .map(|i| MidValue {
                id: MidValueId::from_index(i),
                origin: ValueId::from_index(i),
                storage_group: MidValueId::from_index(if i == 1 { 0 } else { i }),
                owners: crate::tensor::OwnerMap::default(),
                tensor_type: TensorType::new([64], Precision::F16, Layout::logical_linear(2, 4)),
            })
            .collect::<Vec<_>>();
        values[4].storage_group = MidValueId::from_index(2);
        values[4].tensor_type =
            TensorType::new([8192], Precision::F16, Layout::logical_linear(8, 4));
        let parameters = [
            MidValueId::from_index(0),
            MidValueId::from_index(1),
            MidValueId::from_index(2),
            MidValueId::from_index(3),
        ];
        assign_parameter_tiles(&mut values, &parameters, &BTreeMap::new(), 8).unwrap();
        assert_eq!(values[0].owners, values[1].owners);
        assert_eq!(values[2].owners, values[4].owners);
        assert_ne!(values[0].owners, values[2].owners);
        let offsets = values
            .iter()
            .map(|v| v.owners.rotation())
            .collect::<Vec<_>>();
        values[4].tensor_type.shape = TensorShape(vec![16384]);
        assign_parameter_tiles(&mut values, &parameters, &BTreeMap::new(), 8).unwrap();
        assert_eq!(
            offsets,
            values
                .iter()
                .map(|v| v.owners.rotation())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rotations_match_full_load_array_scoring() {
        let mut random = fastrand::Rng::with_seed(0x0074_696c_6573);
        for _ in 0..1000 {
            let tiles = random.usize(1..=64);
            let loads = (0..tiles).map(|_| random.u64(0..4096)).collect::<Vec<_>>();
            let bytes = (0..random.usize(0..=tiles))
                .map(|_| random.u64(0..4096))
                .collect::<Vec<_>>();
            let expected = (0..tiles)
                .min_by_key(|&offset| {
                    let mut candidate = loads.clone();
                    for (logical, &bytes) in bytes.iter().enumerate() {
                        let tile = (logical + offset) % tiles;
                        candidate[tile] = candidate[tile].saturating_add(bytes);
                    }
                    (candidate.into_iter().max().unwrap(), offset)
                })
                .unwrap() as u16;
            assert_eq!(balanced_offset(&loads, &bytes), expected);
        }
    }
}
