//! Bind address-independent copy plans to low-level shard identities.

use super::*;

impl LowShard {
    pub(crate) fn storage(&self) -> crate::storage::TensorStorage<'_> {
        crate::storage::TensorStorage {
            format: &self.tensor_type.format,
            extents: &self.extents,
        }
    }
}

pub fn shard_storage_bytes(shard: &LowShard) -> Result<u32, StorageError> {
    crate::storage::storage_bytes(shard.storage())
}

pub fn view_byte_spans(shard: &LowShard, view: &ShardView) -> Result<Vec<ByteSpan>, StorageError> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    crate::storage::physical_byte_spans(shard.storage(), &view.extents)
}

pub fn logical_view_byte_spans(
    shard: &LowShard,
    view: &ShardView,
) -> Result<Vec<ByteSpan>, StorageError> {
    if view.shard != shard.id {
        return Err(StorageError::WrongShard);
    }
    crate::storage::logical_byte_spans(shard.storage(), &view.extents)
}

pub(super) fn append_span_copies(
    shards: &[LowShard],
    source: &ShardView,
    destination: &ShardView,
    tile: u16,
    copies: &mut Vec<(u16, LocalCopy)>,
    order: CopyOrder,
) -> LowLoweringResult<()> {
    let spans = match order {
        CopyOrder::Semantic => logical_view_byte_spans,
        CopyOrder::Physical => view_byte_spans,
    };
    copies.extend(
        LocalCopy::from_spans(
            source.shard,
            destination.shard,
            &spans(&shards[source.shard.index() as usize], source)?,
            &spans(&shards[destination.shard.index() as usize], destination)?,
        )?
        .into_iter()
        .map(|copy| (tile, copy)),
    );
    Ok(())
}
