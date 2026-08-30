//! Cheap parallel-reduction grid filtering before concrete layout expansion.

use crate::layout::{AMP_COLUMN_MICRO, TensorType};
use crate::metrics::{CostEstimate, MemoryPeaks, RegionMetrics};
use crate::operator::{GemmBlockShape, GemmGrid, GemmOrientation, Precision};
use ipu_target::hardware::HardwareTarget;

pub(crate) fn parallel_reduction_preselection_metrics(
    target: HardwareTarget,
    block: GemmBlockShape,
    grid: GemmGrid,
    orientation: GemmOrientation,
    inputs: &[TensorType],
    output_precision: Precision,
) -> Option<RegionMetrics> {
    let [left, right] = inputs else { return None };
    if left.shape.0.len() < 2 || right.shape.0.len() < 2 {
        return None;
    }
    let outer_rows = left.shape.0[..left.shape.0.len() - 2]
        .iter()
        .fold(1u64, |product, &extent| {
            product.saturating_mul(u64::from(extent))
        });
    let [physical_left, physical_right] = orientation.physical_order([left, right]);
    let logical_rows = physical_left.shape.0[orientation
        .row_axis()
        .resolve(physical_left.shape.0.len())
        .ok()?];
    let local_rows = logical_rows.div_ceil(u32::from(grid.rows));
    let local_columns = block.output_columns.div_ceil(AMP_COLUMN_MICRO);
    let local_inner = block.inner.div_ceil(AMP_COLUMN_MICRO);
    let costs = target.costs();
    let compute = u64::from(local_columns)
        .saturating_mul(u64::from(local_inner))
        .saturating_mul(
            outer_rows
                .saturating_mul(u64::from(local_rows))
                .saturating_mul(4)
                .saturating_add(costs.amp_grid_search_setup_cycles),
        );
    let communication = u64::from(local_columns)
        .saturating_mul(u64::from(local_inner))
        .saturating_add(u64::from(local_rows).saturating_mul(u64::from(local_inner)))
        .saturating_add(
            u64::from(local_rows)
                .saturating_mul(u64::from(local_columns))
                .saturating_mul(u64::from(grid.inner.saturating_sub(1))),
        );
    let left_precision = physical_left.format.precision;
    let right_precision = physical_right.format.precision;
    let left_bytes = outer_rows
        .saturating_mul(u64::from(local_rows))
        .saturating_mul(u64::from(local_inner))
        .saturating_mul(u64::from(AMP_COLUMN_MICRO))
        .saturating_mul(left_precision.bytes());
    let right_bytes = u64::from(local_columns)
        .saturating_mul(u64::from(AMP_COLUMN_MICRO))
        .saturating_mul(u64::from(local_inner))
        .saturating_mul(u64::from(AMP_COLUMN_MICRO))
        .saturating_mul(right_precision.bytes());
    let partial_bytes = outer_rows
        .saturating_mul(u64::from(local_rows))
        .saturating_mul(u64::from(local_columns))
        .saturating_mul(u64::from(AMP_COLUMN_MICRO))
        .saturating_mul(output_precision.bytes());
    let compute_interleaved = left_bytes
        .saturating_add(right_bytes)
        .saturating_add(partial_bytes);
    let reduction_standard = partial_bytes.saturating_mul(3);
    Some(RegionMetrics {
        cost: CostEstimate {
            cycles: compute.saturating_add(communication),
            exchange_cycles: communication,
            ..CostEstimate::default()
        },
        memory: MemoryPeaks {
            standard: reduction_standard,
            interleaved: compute_interleaved,
            total: compute_interleaved.max(reduction_standard.saturating_add(partial_bytes)),
            ..MemoryPeaks::default()
        },
    })
}
