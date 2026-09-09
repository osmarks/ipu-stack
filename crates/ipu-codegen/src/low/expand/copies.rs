//! Bind address-independent copy plans to low-level shard identities.

use super::*;

impl BlockValue {
    pub(crate) fn storage(&self) -> crate::storage::TensorStorage<'_> {
        crate::storage::TensorStorage {
            format: &self.tensor_type.format,
            extents: &self.extents,
        }
    }
}

pub fn shard_storage_bytes(shard: &BlockValue) -> Result<u32, StorageError> {
    crate::storage::storage_bytes(shard.storage())
}

#[cfg(test)]
pub fn view_byte_spans(
    shard: &BlockValue,
    view: &ShardView,
) -> Result<Vec<ByteSpan>, StorageError> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    crate::storage::physical_byte_spans(shard.storage(), &view.extents)
}

pub fn logical_view_byte_spans(
    shard: &BlockValue,
    view: &ShardView,
) -> Result<Vec<ByteSpan>, StorageError> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    crate::storage::logical_byte_spans(shard.storage(), &view.extents)
}

pub(crate) fn view_byte_traversal(
    shard: &BlockValue,
    view: &ShardView,
    order: CopyOrder,
) -> Result<crate::storage::ByteTraversal, StorageError> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    match order {
        CopyOrder::Panels => crate::storage::panel_byte_traversal(shard.storage(), &view.extents),
        CopyOrder::Semantic | CopyOrder::Physical => crate::storage::byte_traversal(
            shard.storage(),
            &view.extents,
            order == CopyOrder::Physical,
        ),
    }
}

pub(super) fn append_span_copies(
    shards: &[BlockValue],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
    order: CopyOrder,
) -> ExpansionResult<()> {
    let source_spans = view_byte_traversal(&shards[source.shard.index() as usize], source, order)?;
    let destination_spans = view_byte_traversal(
        &shards[destination.shard.index() as usize],
        destination,
        order,
    )?;
    copies.extend(
        LocalCopy::from_traversals(
            source.shard,
            destination.shard,
            &source_spans,
            &destination_spans,
        )?
        .into_iter()
        .map(|copy| (tile, copy)),
    );
    Ok(())
}
