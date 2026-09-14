//! Pack smaller panels on additional owners, then transfer packed storage.

use crate::kernel::TileKernelSpec;
use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
    OperandIndexing,
};
use crate::tensor::{
    AxisTiling, BlockMajorOrder, ElementOrder, Layout, Padding, Precision, TensorAxis,
    TensorTiling, TensorType,
};

impl MidProgram {
    pub(crate) fn with_distributed_packing(&self, rows: u16) -> Option<Self> {
        let mut result = self.clone();
        if !distribute_region(
            &mut result.operations,
            &mut result.values,
            self.tile_count,
            rows,
        ) {
            return None;
        }
        result.refresh_estimates()?;
        Some(result)
    }
}

fn packing_layout(tensor: &TensorType, capacity: u16, block_rows: u16) -> Option<Layout> {
    let ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block,
        column_block,
    }) = tensor.format.layout.order
    else {
        return None;
    };
    if block_rows >= row_block || column_block != 16 || tensor.format.layout.tiling.replicas != 1 {
        return None;
    }
    let rank = tensor.shape.0.len();
    let row_axis = rank.checked_sub(2)?;
    let parts = u16::try_from(tensor.shape.0[row_axis].div_ceil(u32::from(block_rows))).ok()?;
    let original = &tensor.format.layout.tiling;
    let mut axes = Vec::new();
    let mut tiles = 1u16;
    for index in (0..rank).rev() {
        let axis = if index == row_axis {
            AxisTiling::new(
                TensorAxis::FromStart(index as u16),
                parts,
                u32::from(block_rows),
                Padding::Zero,
            )
        } else if let Some(axis) = original
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(rank) == Ok(index))
        {
            *axis
        } else {
            continue;
        };
        axes.push(axis.with_tile_stride(tiles));
        tiles = tiles.checked_mul(axis.partitions)?;
    }
    if tiles <= original.tile_count || tiles > capacity {
        return None;
    }
    let mut layout = Layout::row_major(TensorTiling {
        tile_count: tiles,
        replicas: 1,
        axes,
    });
    layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
        row_block: block_rows,
        column_block,
    });
    if layout.resolve(&tensor.shape).ok()?.has_empty_shards() {
        return None;
    }
    Some(layout)
}

fn distribute_region(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    capacity: u16,
    rows: u16,
) -> bool {
    let mut result = Vec::new();
    let mut changed = false;
    for mut operation in std::mem::take(operations) {
        if let MidOperationKind::Repeat(repeat) = &mut operation.kind {
            changed |= distribute_region(&mut repeat.body.operations, values, capacity, rows);
        }
        if let MidOperationKind::Copy { .. } = &operation.kind
            && let ([input], [output]) = (operation.inputs.as_slice(), operation.results.as_slice())
            && values[input.index() as usize].tensor_type.format.precision == Precision::F16
            && values[input.index() as usize]
                .tensor_type
                .format
                .layout
                .order
                == ElementOrder::RowMajor
            && values[output.index() as usize].tensor_type.format.precision == Precision::F16
        {
            let target = values[output.index() as usize].clone();
            if let Some(layout) = packing_layout(&target.tensor_type, capacity, rows) {
                let start = values.len();
                let packed = MidValueId(start as u32 + 1);
                let logical = MidValueId(start as u32);
                let mut tensor = target.tensor_type.clone();
                tensor.format.layout = layout.clone();
                let mut row_major = tensor.clone();
                row_major.format.layout.order = ElementOrder::RowMajor;
                for (id, tensor_type) in [(logical, row_major.clone()), (packed, tensor)] {
                    values.push(MidValue {
                        id,
                        tensor_type,
                        storage_group: id,
                        owners: target.owners.clone(),
                        ..target
                    });
                }
                let pack = MidOperation {
                    inputs: vec![logical],
                    results: vec![packed],
                    source: operation.source,
                    kind: MidOperationKind::Compute(Compute::Kernel {
                        kernel: TileKernelSpec::Rearrange {
                            from: row_major.format.layout,
                            to: layout,
                        },
                        operands: vec![OperandIndexing::Elementwise { result: 0 }],
                        output_aliases: Vec::new(),
                    }),
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                let transfer = MidOperation {
                    inputs: vec![packed],
                    results: vec![*output],
                    source: operation.source,
                    kind: MidOperationKind::Copy {
                        policy: crate::CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Automatic,
                        mapping: CoordinateMapping::default(),
                        reuse_local: true,
                    },
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                };
                operation.results = vec![logical];
                result.extend([operation, pack, transfer]);
                changed = true;
                continue;
            }
        }
        result.push(operation);
    }
    *operations = result;
    changed
}

#[cfg(test)]
mod tests {
    use crate::graph::{GraphInputKind, ValueId};
    use crate::mid::MidInput;

    use crate::tensor::{AxisFactorView, MemoryClass};

    use super::*;
    #[test]
    fn distributed_panels_retile_without_unpacking_the_packed_intermediate() {
        for view in [false, true] {
            let source = TensorType::new(
                if view {
                    vec![1, 729, 144]
                } else {
                    vec![2, 729, 72]
                },
                Precision::F16,
                Layout::row_sharded(64),
            );
            let target = TensorType::new(
                [2, 729, 72],
                Precision::F16,
                Layout {
                    order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                        row_block: 768,
                        column_block: 16,
                    }),
                    memory_class: MemoryClass::Ipu21Standard,
                    tiling: TensorTiling {
                        tile_count: 10,
                        replicas: 1,
                        axes: vec![
                            AxisTiling::new(TensorAxis::FromEnd(1), 5, 16, Padding::Zero)
                                .with_tile_stride(1),
                            AxisTiling::new(TensorAxis::FromEnd(2), 1, 768, Padding::Zero)
                                .with_tile_stride(5),
                            AxisTiling::new(TensorAxis::FromEnd(3), 2, 1, Padding::Reject)
                                .with_tile_stride(5),
                        ],
                    },
                },
            );
            let values = [source, target]
                .into_iter()
                .enumerate()
                .map(|(index, tensor_type)| {
                    let id = MidValueId(index as u32);
                    MidValue {
                        id,
                        tensor_type,
                        origin: ValueId::from_index(index as u32),
                        storage_group: id,
                        owners: crate::tensor::OwnerMap::default(),
                    }
                })
                .collect();
            let program = MidProgram {
                tile_count: 64,
                values,
                inputs: vec![MidInput {
                    name: "x".into(),
                    kind: GraphInputKind::Host,
                    value: MidValueId(0),
                }],
                outputs: vec![MidValueId(1)],
                operations: vec![MidOperation {
                    source: None,
                    inputs: vec![MidValueId(0)],
                    results: vec![MidValueId(1)],
                    kind: MidOperationKind::Copy {
                        policy: crate::CopyPolicy::Automatic,
                        packing: crate::PackingPolicy::Automatic,
                        mapping: CoordinateMapping {
                            offsets: vec![],
                            view: view.then_some(AxisFactorView::new(2, 0, 2)),
                        },
                        reuse_local: true,
                    },
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                }],
                ..MidProgram::default()
            };
            let candidates = [32, 64, 128, 256]
                .into_iter()
                .filter_map(|rows| program.with_distributed_packing(rows))
                .collect::<Vec<_>>();
            assert!(!candidates.is_empty() && candidates.len() <= 4);
            for packed in candidates {
                let low = crate::lower_to_tiles(
                    &crate::low::expand::expand_tiles(&packed, false).unwrap(),
                    false,
                );
                let mut packs = 0;
                for run in &low.kernel_runs {
                    if let TileKernelSpec::Rearrange { from, to } = &run.kernel {
                        assert_eq!(from.order, ElementOrder::RowMajor);
                        assert!(matches!(
                            to.order,
                            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                                row_block: 32..=256,
                                ..
                            })
                        ));
                        run.call().unwrap();
                        packs += 1;
                    }
                }
                assert!(packs > 10 && packs <= 64);
            }
        }
    }
}
