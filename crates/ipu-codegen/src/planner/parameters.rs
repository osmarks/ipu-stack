//! Compact resident defaults from the previous baseline planner. Keep small
//! parameters in transfer-sized chunks rather than spreading a few bytes over
//! every tile. These reservations are not lower bounds or operand constraints.

use crate::{Layout, PipelineConfig, Precision, TensorFormat, TensorShape};

pub(super) fn default_format(shape: &TensorShape, config: &PipelineConfig) -> TensorFormat {
    // The current constructors support FP16 elementwise operations. Native
    // packed-matrix homes belong with the future GEMM candidate constructor.
    let precision = Precision::F16;
    let bytes = shape.elements().saturating_mul(precision.bytes());
    let limit = config
        .tile_memory_budget_bytes
        .min(config.target.planned_data_bytes())
        .saturating_sub(config.standard_memory_reservation_bytes);
    let required = bytes.div_ceil(limit.max(1));
    let owners = bytes
        .div_ceil(256)
        .max(required)
        .max(1)
        .min(u64::from(config.tile_count));
    TensorFormat {
        precision,
        layout: Layout::logical_linear(owners as u16, 1),
    }
}
