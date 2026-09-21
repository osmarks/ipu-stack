//! Resident defaults follow the first consumer's storage conventions. They are
//! reservations, not operand constraints. Aligned flat storage remains an
//! alternative when whole-row ownership concentrates too much memory.

use super::{PlanningError, PlanningResult};
use crate::graph::{HighGraph, OperationKind, ValueId};
use crate::{Layout, PipelineConfig, Precision, TensorFormat, TensorShape};

pub(super) fn default_format(
    high: &HighGraph,
    value: ValueId,
    config: &PipelineConfig,
) -> PlanningResult<TensorFormat> {
    let consumer = high
        .operations()
        .iter()
        .find(|op| high.operation_inputs(op).any(|id| id == value));
    match consumer.map(|op| &op.kind) {
        None | Some(OperationKind::Add | OperationKind::Gelu) => {}
        // Do not guess row-major storage for a future packed-kernel consumer.
        Some(_) => {
            return Err(PlanningError::Unimplemented(
                "parameter format for this consumer",
            ));
        }
    }
    let shape = high
        .value_shape(value)
        .ok_or(PlanningError::InvalidFragment("unknown parameter"))?;
    let mut format = compact_format(shape, config);
    if shape.0.len() >= 2 {
        let rows = shape.0[shape.0.len() - 2];
        format.layout = Layout::row_sharded(rows.min(u32::from(config.tile_count)) as u16);
    }
    Ok(format)
}

pub(super) fn compact_format(shape: &TensorShape, config: &PipelineConfig) -> TensorFormat {
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
    // Linear ownership cannot represent a padded logical tail. Vectors can
    // use ordinary axis partitioning, which can pad the final four-FP16 block.
    let mut layout = if shape.0.len() == 1 {
        Layout::row_major(crate::TensorTiling::sharded(
            crate::TensorAxis::FromEnd(1),
            owners as u16,
        ))
    } else {
        let grain = if shape.elements().is_multiple_of(4) {
            4
        } else {
            1
        };
        Layout::logical_linear(owners as u16, grain)
    };
    if shape.0.len() == 1 {
        layout.tiling.axes[0].block_size = 4;
        layout.tiling.axes[0].padding_multiple = 4;
        layout.tiling.axes[0].padding = crate::Padding::Zero;
    }
    TensorFormat { precision, layout }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_rows_or_aligned_contiguous_vector_intervals() {
        let mut rng = fastrand::Rng::with_seed(0xdefa_017);
        for _ in 0..128 {
            let config = PipelineConfig::new(ipu_target::Target::Ipu21, rng.u16(1..=64));
            for shape in [
                vec![rng.u32(1..=1024)],
                vec![rng.u32(1..=100), rng.u32(1..=200)],
            ] {
                let mut high = HighGraph::new();
                let parameter = high.parameter("p", shape.clone()).unwrap();
                let output = high.gelu(parameter).unwrap();
                high.set_outputs([output]).unwrap();
                let format = default_format(&high, parameter, &config).unwrap();
                let shards = format
                    .layout
                    .resolve(&TensorShape(shape.clone()))
                    .unwrap()
                    .shard_extents()
                    .unwrap();
                let mut end = 0;
                for (_, axes) in shards {
                    let partition = &axes[0];
                    assert_eq!(partition.start, end);
                    end = partition.logical_end;
                    if shape.len() == 1 {
                        assert_eq!(partition.start % 4, 0);
                        assert_eq!(partition.physical_end % 4, 0);
                    } else {
                        assert_eq!(
                            (axes[1].start, axes[1].logical_end, axes[1].physical_end),
                            (0, shape[1], shape[1])
                        );
                    }
                }
                assert_eq!(end, shape[0]);
            }
        }
    }
}
