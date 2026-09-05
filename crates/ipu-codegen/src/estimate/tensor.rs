//! Tensor storage geometry used by memory and cycle estimates.

use super::*;

pub(crate) fn physical_elements(shape: &TensorShape, layout: &Layout) -> u64 {
    layout
        .resolve(shape)
        .map_or(u64::MAX, |resolved| resolved.physical_elements())
}

pub(crate) fn maximum_shard_bytes(tensor: &TensorType) -> u64 {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .map_or(u64::MAX, |resolved| {
            resolved
                .maximum_tile_elements()
                .saturating_mul(tensor.format.precision.bytes())
        })
}

pub(crate) fn maximum_axis_shard_extent(tensor: &TensorType, axis: usize) -> u64 {
    tensor
        .format
        .layout
        .resolve(&tensor.shape)
        .ok()
        .and_then(|resolved| {
            resolved
                .axes()?
                .get(axis)
                .map(|axis| u64::from(axis.maximum_extent()))
        })
        .unwrap_or(u64::MAX)
}
