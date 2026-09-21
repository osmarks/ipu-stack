//! Resident defaults follow the first consumer's storage conventions. They are
//! reservations, not operand constraints. Aligned flat storage remains an
//! alternative when whole-row ownership concentrates too much memory.

use super::{PlanningError, PlanningResult};
use crate::graph::{HighGraph, OperationKind, ValueId};
use crate::{Layout, PipelineConfig, Precision, TensorFormat, TensorShape};

pub(super) fn initial_format(
    high: &HighGraph,
    input: &crate::graph::GraphInput,
    layout: Option<&Layout>,
    config: &PipelineConfig,
) -> PlanningResult<TensorFormat> {
    let mut format = match config.inputs.get(&input.value) {
        Some(format) => format.clone(),
        None if input.kind == crate::graph::GraphInputKind::Parameter => {
            default_format(high, input.value, config)?
        }
        None => return Err(PlanningError::UnassignedLayout(input.value)),
    };
    if let Some(layout) = layout {
        format.layout = layout.clone();
    }
    Ok(format)
}

pub(super) fn default_format(
    high: &HighGraph,
    value: ValueId,
    config: &PipelineConfig,
) -> PlanningResult<TensorFormat> {
    let consumer = high
        .operations()
        .iter()
        .find(|op| high.operation_inputs(op).any(|id| id == value));
    let shape = high
        .value_shape(value)
        .ok_or(PlanningError::InvalidFragment("unknown parameter"))?;
    match consumer.map(|op| &op.kind) {
        None | Some(OperationKind::Add | OperationKind::Gelu) => {}
        Some(OperationKind::Gemm(options)) => {
            let left = consumer.unwrap().inputs[0] == value;
            return Ok(super::gemm::parameter_format(
                shape,
                left,
                if left {
                    options.transpose_left
                } else {
                    options.transpose_right
                },
                config,
            ));
        }
        // Do not guess row-major storage for a future packed-kernel consumer.
        Some(_) => {
            return Err(PlanningError::Unimplemented(
                "parameter format for this consumer",
            ));
        }
    }
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
    // Four FP16 elements permit dense 64-bit local copies. Only the final
    // allocation receives tail padding, even for a multidimensional tensor.
    let mut layout = Layout::logical_linear(owners as u16, 4);
    layout.tiling.axes[0].padding = crate::Padding::Zero;
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
