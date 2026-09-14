//! Ordinary block allocations, ownership lookup, and views.

use super::*;

impl TileGraphBuilder {
    pub(super) fn storage_root(&self, shard: BlockValueId) -> BlockValueId {
        crate::storage_root(&self.shards, shard)
    }

    pub(super) fn push_shard(&mut self, mut shard: BlockValue) -> ExpansionResult<BlockValueId> {
        let id =
            BlockValueId(u32::try_from(self.shards.len()).map_err(|_| ExpansionError::IdOverflow)?);
        shard.id = id;
        self.shards.push(shard);
        Ok(id)
    }

    pub(super) fn value_shards(&self, value: MidValueId) -> ExpansionResult<&[BlockValueId]> {
        self.canonical
            .get(value.index() as usize)
            .filter(|shards| !shards.is_empty())
            .map(Vec::as_slice)
            .ok_or(ExpansionError::UnknownValue(value))
    }

    pub(super) fn local_shard(
        &self,
        value: MidValueId,
        tile: u16,
    ) -> ExpansionResult<BlockValueId> {
        let shards = self.value_shards(value)?;
        if let Some(&shard) = shards.get(usize::from(tile))
            && self.shards[shard.index() as usize].tile == tile
        {
            return Ok(shard);
        }
        shards
            .iter()
            .copied()
            .find(|shard| self.shards[shard.index() as usize].tile == tile)
            .ok_or(ExpansionError::UnknownValue(value))
    }

    pub(super) fn full_view(&self, shard: BlockValueId) -> ShardView {
        if let Some(view) = self.borrowed_views.get(&shard) {
            return view.clone();
        }
        ShardView {
            shard,
            extents: self.shards[shard.index() as usize].extents.clone(),
        }
    }

    /// Resolve a logical read before inspecting its physical storage. Borrowed
    /// copies preserve coordinates, but their backing allocation can have a
    /// larger extent (and therefore a different row or panel stride).
    pub(super) fn resolve_read_view(&self, view: &mut ShardView) -> ExpansionResult<()> {
        if let Some(source) = self.borrowed_views.get(&view.shard) {
            if view.extents.len() != source.extents.len()
                || view
                    .extents
                    .iter()
                    .zip(&source.extents)
                    .any(|(read, source)| {
                        read.axis != source.axis
                            || read.start < source.start
                            || read.logical_end > source.logical_end
                            || read.physical_end > source.physical_end
                    })
            {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            view.shard = source.shard;
        }
        if self.shards[self.storage_root(view.shard).index() as usize].definition
            == ShardDefinition::Unmaterialized
        {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        Ok(())
    }

    pub(super) fn narrow_view(
        &self,
        shard: BlockValueId,
        ranges: &[(usize, u32, u32)],
    ) -> ExpansionResult<ShardView> {
        let mut view = self.full_view(shard);
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
        self.value_shards(value)?
            .iter()
            .copied()
            .filter(|shard| self.shards[shard.index() as usize].extents == target.extents)
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != target.tile))
            .ok_or(ExpansionError::UnknownValue(value))
    }

    pub(super) fn shard_stride(
        &self,
        shard: BlockValueId,
        alignment: u32,
        access_tail: u32,
    ) -> ExpansionResult<u32> {
        let shard = &self.shards[shard.index() as usize];
        shard_storage_bytes(shard)
            .map_err(|_| ExpansionError::IdOverflow)?
            .checked_add(access_tail)
            .and_then(|bytes| bytes.checked_next_multiple_of(alignment.max(1)))
            .ok_or(ExpansionError::IdOverflow)
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
