//! Logical view geometry, shared by materialization and bounded dispatch slices.

use super::{ShardExtent, TensorShape};

/// Move a factor from one axis into another, preserving the other axes.
/// For example `(split_axis=2, merge_axis=0, factor=H)` maps
/// `[B, R, H*C]` to `[B*H, R, C]`. Axis positions and rank are unrestricted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AxisFactorView {
    pub split_axis: usize,
    pub merge_axis: usize,
    pub factor: u32,
}

impl AxisFactorView {
    pub const fn new(split_axis: usize, merge_axis: usize, factor: u32) -> Self {
        Self {
            split_axis,
            merge_axis,
            factor,
        }
    }

    pub fn output_shape(self, source: &TensorShape) -> Option<TensorShape> {
        let mut shape = source.0.clone();
        if self.factor == 0
            || self.split_axis == self.merge_axis
            || self.merge_axis >= shape.len()
            || !shape.get(self.split_axis)?.is_multiple_of(self.factor)
        {
            return None;
        }
        shape[self.split_axis] /= self.factor;
        shape[self.merge_axis] = shape[self.merge_axis].checked_mul(self.factor)?;
        Some(TensorShape(shape))
    }

    /// A rectangular slice with a singleton merged axis has a rectangular
    /// preimage. Larger slices must be divided at merged-axis coordinates.
    pub fn map_slice(
        self,
        source_shape: &TensorShape,
        output_shape: &TensorShape,
        output: &[(u32, u32)],
    ) -> Option<DeferredSliceMapping> {
        if self.output_shape(source_shape).as_ref() != Some(output_shape)
            || output.len() != output_shape.0.len()
            || output
                .iter()
                .zip(&output_shape.0)
                .any(|(&(start, end), &size)| start >= end || end > size)
        {
            return None;
        }
        let (start, end) = output[self.merge_axis];
        if end != start.checked_add(1)? {
            return None;
        }
        let base = (start % self.factor).checked_mul(output_shape.0[self.split_axis])?;
        let mut source_ranges = output.to_vec();
        source_ranges[self.merge_axis] = (start / self.factor, start / self.factor + 1);
        source_ranges[self.split_axis] = (
            base.checked_add(output[self.split_axis].0)?,
            base.checked_add(output[self.split_axis].1)?,
        );
        Some(DeferredSliceMapping {
            source_ranges,
            destination_source_axes: (0..output.len())
                .filter(|&axis| axis != self.merge_axis)
                .collect(),
        })
    }

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
        let base = mapping.source_ranges[self.split_axis].0 - output[self.split_axis].start;
        let extents = mapping
            .source_ranges
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredSliceMapping {
    pub source_ranges: Vec<(u32, u32)>,
    /// Source axes retained in the slice buffer; the singleton merged axis
    /// selects the slice but occupies no dimension in that buffer.
    pub destination_source_axes: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_views_cover_source_once_for_every_axis_pair() {
        for rank in 2..=5 {
            for split in 0..rank {
                for merge in 0..rank {
                    if split == merge {
                        continue;
                    }
                    let mut shape = vec![2; rank];
                    shape[split] = 6;
                    let source = TensorShape(shape);
                    let view = AxisFactorView::new(split, merge, 3);
                    let output = view.output_shape(&source).unwrap();
                    let mut covered = std::collections::BTreeSet::new();
                    for linear in 0..output.0.iter().product::<u32>() {
                        let mut index = linear;
                        let mut point = vec![(0, 0); rank];
                        for axis in (0..rank).rev() {
                            let coordinate = index % output.0[axis];
                            index /= output.0[axis];
                            point[axis] = (coordinate, coordinate + 1);
                        }
                        let mapped = view.map_slice(&source, &output, &point).unwrap();
                        let source_index = mapped.source_ranges.iter().zip(&source.0).fold(
                            0,
                            |index, (&(start, end), &size)| {
                                assert_eq!(end, start + 1);
                                assert!(end <= size);
                                index * size + start
                            },
                        );
                        assert!(covered.insert(source_index), "two output elements alias");
                    }
                    assert_eq!(covered.len() as u32, source.0.iter().product::<u32>());
                }
            }
        }
    }

    #[test]
    fn factor_views_reject_invalid_shapes_and_nonrectangular_slices() {
        let source = TensorShape(vec![2, 4, 12]);
        for view in [
            AxisFactorView::new(2, 0, 0),
            AxisFactorView::new(2, 2, 3),
            AxisFactorView::new(3, 0, 3),
            AxisFactorView::new(2, 3, 3),
            AxisFactorView::new(2, 0, 5),
        ] {
            assert!(view.output_shape(&source).is_none());
        }
        let view = AxisFactorView::new(2, 0, 3);
        let output = view.output_shape(&source).unwrap();
        for slice in [
            vec![(0, 2), (0, 4), (0, 4)],
            vec![(6, 7), (0, 4), (0, 4)],
            vec![(0, 1), (2, 1), (0, 4)],
            vec![(0, 1), (0, 4), (0, 5)],
        ] {
            assert!(view.map_slice(&source, &output, &slice).is_none());
        }
        assert!(
            view.output_shape(&TensorShape(vec![u32::MAX, 4, 12]))
                .is_none()
        );
    }
}
