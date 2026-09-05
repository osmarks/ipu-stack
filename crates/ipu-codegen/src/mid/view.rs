//! Storage-extent adapter for semantic view geometry.

use super::{AxisFactorView, ShardExtent, TensorShape};

impl AxisFactorView {
    pub(crate) fn source_extents(
        self,
        source_shape: &TensorShape,
        output_shape: &TensorShape,
        output: &[ShardExtent],
    ) -> Option<(Vec<ShardExtent>, u32)> {
        let ranges = output
            .iter()
            .map(|extent| (extent.start, extent.logical_end))
            .collect::<Vec<_>>();
        let mapping = self.map_slice(source_shape, output_shape, &ranges)?;
        let base = mapping[self.split_axis].0 - output[self.split_axis].start;
        let extents = mapping
            .into_iter()
            .enumerate()
            .map(|(axis, (start, end))| {
                Some(ShardExtent {
                    axis: u16::try_from(axis).ok()?,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some((extents, base))
    }
}
