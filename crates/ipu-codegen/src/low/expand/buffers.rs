//! Ordinary block allocations, ownership lookup, and views.

use super::*;
use crate::mid::MidOperationKind;

impl TileGraphBuilder {
    pub(super) fn bind_kernel(
        &mut self,
        provenance: WorkProvenance,
        kernel: MidOperationKind,
        inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
    ) -> ExpansionResult<KernelRun> {
        Ok(KernelRun::bind(
            provenance,
            kernel,
            inputs,
            outputs,
            &self.shards,
            &mut self.kernel_metadata,
        )?)
    }

    pub(super) fn push_shard(&mut self, mut shard: BlockValue) -> ExpansionResult<BlockValueId> {
        let id =
            BlockValueId(u32::try_from(self.shards.len()).map_err(|_| ExpansionError::IdOverflow)?);
        shard.id = id;
        self.shards.push(shard);
        Ok(id)
    }

    pub(super) fn value_views(&self, value: MidValueId) -> ExpansionResult<&[ShardView]> {
        self.bindings
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(ExpansionError::UnknownValue(value))
    }

    pub(super) fn full_view(&self, shard: BlockValueId) -> ShardView {
        ShardView {
            shard,
            extents: self.shards[shard.index() as usize].extents.clone(),
        }
    }

    /// Bind a whole allocation when a write/region ABI needs its exact strides.
    /// Read consumers use value_views and keep their logical selection.
    pub(super) fn allocation_shards(
        &self,
        value: MidValueId,
    ) -> ExpansionResult<Vec<BlockValueId>> {
        self.value_views(value)?
            .iter()
            .map(|view| {
                if view.extents != self.shards[view.shard.index() as usize].extents {
                    return Err(ExpansionError::InvalidOperatorPlan);
                }
                Ok(view.shard)
            })
            .collect()
    }

    pub(super) fn narrow_view(
        &self,
        source: &ShardView,
        ranges: &[(usize, u32, u32)],
    ) -> ExpansionResult<ShardView> {
        let mut view = source.clone();
        for &(axis, start, end) in ranges {
            let extent = view
                .extents
                .get_mut(axis)
                .ok_or(ExpansionError::InvalidOperatorPlan)?;
            if start < extent.start || end > extent.physical_end || start >= end {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            extent.start = start;
            extent.physical_end = end;
            extent.logical_end = end.min(extent.logical_end).max(start);
        }
        Ok(view)
    }

    pub(super) fn alias_shard(&mut self, shard: BlockValueId, target: BlockValueId) {
        self.shards[shard.index() as usize].definition = ShardDefinition::Alias(target);
    }

    pub(super) fn corresponding_shard(
        &self,
        value: MidValueId,
        target: BlockValueId,
    ) -> ExpansionResult<BlockValueId> {
        let target = &self.shards[target.index() as usize];
        self.allocation_shards(value)?
            .into_iter()
            .filter(|shard| self.shards[shard.index() as usize].extents == target.extents)
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != target.tile))
            .ok_or(ExpansionError::UnknownValue(value))
    }

    pub(super) fn push_packed_buffer(
        &mut self,
        tile: u16,
        elements: u32,
        precision: Precision,
        definition: ShardDefinition,
    ) -> ExpansionResult<BlockValueId> {
        self.push_shard(BlockValue {
            id: BlockValueId(0),
            tile,
            tensor_type: TensorType::new(
                [elements],
                precision,
                Layout::row_major(TensorTiling::replicated(1)),
            ),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: elements,
                physical_end: elements,
            }],
            definition,
        })
    }
}
