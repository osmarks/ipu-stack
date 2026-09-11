//! Distributed feature statistics followed by local affine normalization.
use super::*;

impl Builder {
    pub(super) fn layernorm(&mut self, output: &TensorType, parts: u16) -> Option<MidValueId> {
        let rank = output.shape.0.len();
        let width = *output.shape.0.last()?;
        if parts < 2 || !width.is_multiple_of(u32::from(parts) * 4) {
            return None;
        }
        let mut inputs = Vec::new();
        for index in 0..3 {
            let id = MidValueId(index);
            let mut tensor = self.tensor(id).clone();
            tensor.format.layout.tiling = pointwise_input_tiling(&tensor, output)?;
            inputs.push(self.copy(id, tensor, vec![]));
        }
        let mut moments = output.clone();
        moments.format.precision = Precision::F32;
        moments.shape.0.pop();
        moments.shape.0.extend([u32::from(parts), 2]);
        for dim in &mut moments.format.layout.tiling.axes {
            let index = dim.axis.resolve(rank).ok()?;
            dim.axis = TensorAxis::FromStart(index as u16);
            if index + 1 == rank {
                dim.block_size = 1;
            }
        }
        let partials = self.compute(
            vec![inputs[0]],
            moments.clone(),
            TileKernelSpec::LayerNormMoments,
            None,
            None,
            vec![],
        );
        // Replicate only the small (mean, variance) vectors;
        // features and affine parameters remain partitioned.
        moments
            .format
            .layout
            .tiling
            .axes
            .retain(|axis| axis.axis != TensorAxis::FromStart((rank - 1) as u16));
        moments.format.layout.tiling.replicas *= parts;
        let complete = self.copy(partials, moments, vec![]);
        inputs.push(complete);
        Some(self.compute(
            inputs,
            output.clone(),
            TileKernelSpec::LayerNormApply { parts },
            None,
            None,
            vec![],
        ))
    }
}
