//! Pointwise implementation construction and broadcast views.

use super::*;

impl TileGraphBuilder {
    pub(super) fn broadcast_view(
        &self,
        source: BlockValueId,
        output: BlockValueId,
    ) -> Option<ShardView> {
        self.broadcast_view_for_extents(
            source,
            output,
            &self.shards[output.index() as usize].extents,
        )
    }

    pub(super) fn broadcast_view_for_extents(
        &self,
        source: BlockValueId,
        output: BlockValueId,
        output_extents: &[ShardExtent],
    ) -> Option<ShardView> {
        let source_shard = &self.shards[source.index() as usize];
        let output_shard = &self.shards[output.index() as usize];
        let source_rank = source_shard.extents.len();
        let output_rank = output_shard.extents.len();
        if source_rank > output_rank {
            return None;
        }
        let offset = output_rank - source_rank;
        let mut extents = source_shard.extents.clone();
        for (axis, extent) in extents.iter_mut().enumerate() {
            let dimension = source_shard.tensor_type.shape.0[axis];
            if dimension == 1 {
                if extent.start != 0 || extent.logical_end == 0 {
                    return None;
                }
                extent.start = 0;
                extent.logical_end = 1;
                extent.physical_end = 1;
            } else {
                let required = output_extents[offset + axis];
                if extent.start > required.start || extent.logical_end < required.logical_end {
                    return None;
                }
                extent.start = required.start;
                extent.logical_end = required.logical_end;
                extent.physical_end = required.logical_end;
            }
        }
        Some(ShardView {
            shard: source,
            extents,
        })
    }
}
