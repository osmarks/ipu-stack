//! Early fragmentation of compatible rectangular ownership grids. This walks
//! axis partitions, never their Cartesian product of tiles or byte spans.

use super::*;
use crate::{AmpOrder, ShardExtent};

type Bounds = Vec<(u32, u32)>;

/// Counts physical runs including local intersections (a conservative estimate
/// before tile ownership is matched). Replicas do not multiply multicast sends.
/// Unsupported mappings retain the caller's coarse estimate.
pub(super) fn grid_fragments(input: &TensorType, output: &TensorType) -> Option<u64> {
    if input.shape != output.shape
        || input.format.precision != output.format.precision
        || input.format.layout.order != output.format.layout.order
    {
        return None;
    }
    let axes = |tensor: &TensorType| -> Option<Vec<Bounds>> {
        let resolved = tensor.format.layout.resolve(&tensor.shape).ok()?;
        Some(
            resolved
                .axes()?
                .iter()
                .map(|axis| {
                    axis.partitions()
                        .iter()
                        .map(|extent: &ShardExtent| (extent.start, extent.physical_end))
                        .collect::<Bounds>()
                })
                .collect::<Vec<_>>(),
        )
    };
    let mut source = axes(input)?;
    let mut destination = axes(output)?;
    let bytes = input.format.precision.bytes();
    // Whole-shard broadcasts are contiguous in every encoding. Only a change
    // of partition bounds needs the physical axis factorization below.
    if source == destination {
        return Some(endpoint_fragments(&source, &destination, bytes));
    }
    for axes in [&mut source, &mut destination] {
        match input.format.layout.order {
            ElementOrder::RowMajor => {}
            ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output) => {
                let grain = input
                    .format
                    .layout
                    .order
                    .retained_linear_column_grain(input.format.precision)?;
                let columns = axes.pop()?;
                if columns
                    .iter()
                    .any(|&(start, end)| !start.is_multiple_of(grain) || !end.is_multiple_of(grain))
                {
                    return None;
                }
                axes.insert(
                    0,
                    columns
                        .into_iter()
                        .map(|(start, end)| (start / grain, end / grain))
                        .collect(),
                );
                axes.push(vec![(0, grain)]);
            }
            _ => return None,
        }
    }
    Some(
        endpoint_fragments(&source, &destination, bytes).max(endpoint_fragments(
            &destination,
            &source,
            bytes,
        )),
    )
}

fn endpoint_fragments(own: &[Bounds], peer: &[Bounds], element_bytes: u64) -> u64 {
    let max_bytes = u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4;
    let maximum_width = |bounds: &Bounds| {
        bounds
            .iter()
            .map(|(start, end)| u64::from(end - start))
            .max()
            .unwrap_or(0)
    };
    // Equal trailing partitions fill both allocations contiguously. At the
    // first differing axis each intersection becomes a run; preceding axes
    // repeat that pattern with a stride in at least one allocation.
    let Some(split) = own.iter().zip(peer).rposition(|(a, b)| a != b) else {
        return own
            .iter()
            .map(maximum_width)
            .fold(element_bytes, u64::saturating_mul)
            .div_ceil(max_bytes);
    };
    let trailing_bytes = own[split + 1..]
        .iter()
        .map(maximum_width)
        .fold(element_bytes, u64::saturating_mul);
    let mut first = 0;
    let mut runs = 0;
    for &(start, end) in &own[split] {
        while first < peer[split].len() && peer[split][first].1 <= start {
            first += 1;
        }
        let intersections = peer[split][first..]
            .iter()
            .take_while(|&&(other_start, _)| other_start < end)
            .map(|&(other_start, other_end)| {
                let width = end.min(other_end).saturating_sub(start.max(other_start));
                (u64::from(width) * trailing_bytes).div_ceil(max_bytes)
            })
            .sum::<u64>();
        runs = runs.max(intersections);
    }
    own[..split]
        .iter()
        .map(maximum_width)
        .fold(runs, u64::saturating_mul)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{TensorStorage, physical_byte_spans};
    use crate::{AxisTiling, Padding, TensorAxis, TensorTiling};

    fn tensor(order: ElementOrder, rows: u16, columns: u16) -> TensorType {
        let mut layout = Layout::row_major(TensorTiling {
            tile_count: rows * columns,
            replicas: 1,
            axes: vec![
                AxisTiling::new(TensorAxis::FromEnd(2), rows, 1, Padding::Reject),
                AxisTiling::new(TensorAxis::FromEnd(1), columns, 32, Padding::Reject),
            ],
        });
        layout.order = order;
        TensorType::new([2, 32, 256], Precision::F16, layout)
    }

    fn expanded_maximum(input: &TensorType, output: &TensorType) -> u64 {
        let sources = input.format.layout.shard_extents(&input.shape).unwrap();
        let destinations = output.format.layout.shard_extents(&output.shape).unwrap();
        let mut outgoing = vec![0; sources.len()];
        let mut incoming = vec![0; destinations.len()];
        for (i, (_, source)) in sources.iter().enumerate() {
            for (j, (_, destination)) in destinations.iter().enumerate() {
                let mut view = source.clone();
                for (extent, other) in view.iter_mut().zip(destination) {
                    extent.start = extent.start.max(other.start);
                    extent.physical_end = extent.physical_end.min(other.physical_end);
                    extent.logical_end = extent.physical_end;
                }
                if view
                    .iter()
                    .any(|extent| extent.start >= extent.physical_end)
                {
                    continue;
                }
                let left = physical_byte_spans(
                    TensorStorage {
                        format: &input.format,
                        extents: source,
                    },
                    &view,
                )
                .unwrap();
                let right = physical_byte_spans(
                    TensorStorage {
                        format: &output.format,
                        extents: destination,
                    },
                    &view,
                )
                .unwrap();
                let mut fragments = 0;
                crate::for_each_copy_span(&left, &right, |_, _, bytes| {
                    fragments +=
                        u64::from(bytes).div_ceil(u64::from(ipu_exchange::MAX_TRANSFER_WORDS) * 4);
                    Ok(())
                })
                .unwrap();
                outgoing[i] += fragments;
                incoming[j] += fragments;
            }
        }
        outgoing.into_iter().chain(incoming).max().unwrap_or(0)
    }

    #[test]
    fn grid_counts_match_expanded_row_and_panel_intersections() {
        for order in [
            ElementOrder::RowMajor,
            ElementOrder::Amp(AmpOrder::Left),
            ElementOrder::Amp(AmpOrder::Output),
        ] {
            for sr in [1, 2, 4] {
                for sc in [1, 2, 4] {
                    for dr in [1, 2, 4] {
                        for dc in [1, 2, 4] {
                            let input = tensor(order, sr, sc);
                            let output = tensor(order, dr, dc);
                            assert_eq!(
                                grid_fragments(&input, &output),
                                Some(expanded_maximum(&input, &output)),
                                "{order:?}: ({sr}, {sc}) -> ({dr}, {dc})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn uneven_and_fp8_grids_match_expanded_spans() {
        for precision in [
            Precision::F8F143 { scale_exponent: -4 },
            Precision::F16,
            Precision::F32,
        ] {
            for order in [
                ElementOrder::RowMajor,
                ElementOrder::Amp(AmpOrder::Left),
                ElementOrder::Amp(AmpOrder::Output),
            ] {
                let mut input = tensor(order, 3, 1);
                let mut output = tensor(order, 2, 4);
                for tensor in [&mut input, &mut output] {
                    tensor.format.precision = precision;
                    tensor.format.layout.tiling.axes[0].padding = Padding::Zero;
                    tensor.shape.0[1] = 31;
                }
                assert_eq!(
                    grid_fragments(&input, &output),
                    Some(expanded_maximum(&input, &output)),
                    "{precision:?} {order:?}"
                );
            }
        }
    }

    #[test]
    fn row_owned_fp8_to_tall_panels_exposes_sub_256_byte_transfers() {
        let input = TensorType::new(
            [2, 729, 1152],
            Precision::F8F143 { scale_exponent: -4 },
            Layout {
                order: ElementOrder::Amp(AmpOrder::Left),
                memory_class: MemoryClass::Ipu21Standard,
                tiling: TensorTiling {
                    tile_count: 1458,
                    replicas: 1,
                    axes: vec![
                        AxisTiling::new(TensorAxis::FromStart(0), 2, 1, Padding::Reject),
                        AxisTiling::new(TensorAxis::FromStart(1), 729, 1, Padding::Reject),
                    ],
                },
            },
        );
        let mut output = input.clone();
        output.format.layout.tiling = TensorTiling {
            tile_count: 60,
            replicas: 1,
            axes: vec![
                AxisTiling::new(TensorAxis::FromStart(1), 10, 2, Padding::Zero),
                AxisTiling::new(TensorAxis::FromStart(2), 6, 192, Padding::Zero),
            ],
        };
        assert_eq!(grid_fragments(&input, &output), Some(888));
        assert_eq!(expanded_maximum(&input, &output), 888);
        // The old payload-only heuristic priced this as 111 fragments.
        assert_eq!(maximum_shard_bytes(&output).div_ceil(256), 111);
        // Replicating each consumer panel adds multicast recipients, not sends.
        output.format.layout.tiling.tile_count *= 24;
        output.format.layout.tiling.replicas = 24;
        assert_eq!(grid_fragments(&input, &output), Some(888));
    }

    #[test]
    fn whole_block_major_shards_remain_contiguous_when_replicated() {
        let input = tensor(
            ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 16,
                column_block: 32,
            }),
            2,
            2,
        );
        let mut output = input.clone();
        output.format.layout.tiling.tile_count *= 8;
        output.format.layout.tiling.replicas = 8;
        assert_eq!(grid_fragments(&input, &output), Some(1));
        let shard = input
            .format
            .layout
            .shard_extents(&input.shape)
            .unwrap()
            .remove(0)
            .1;
        assert_eq!(
            physical_byte_spans(
                TensorStorage {
                    format: &input.format,
                    extents: &shard
                },
                &shard
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn row_and_column_cuts_have_different_fragmentation() {
        let input = tensor(ElementOrder::RowMajor, 1, 1);
        assert_eq!(
            grid_fragments(&input, &tensor(ElementOrder::RowMajor, 4, 1)),
            Some(8)
        );
        assert_eq!(
            grid_fragments(&input, &tensor(ElementOrder::RowMajor, 1, 4)),
            Some(256)
        );
    }
}
