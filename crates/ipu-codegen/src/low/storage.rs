//! Bind selected tensor geometry to low storage, preserving alias interpretation.
//!
//! A shifted FP8 value may be backed by FP16 storage. Its format and strides
//! still come from the selected value; only its byte origin follows aliases.
//! Keep the chain available: Repeat may override an intermediate argument.

use super::{BlockValue, BlockValueId, ShardDefinition, ShardView};
use crate::storage::{ByteSpan, ByteTraversal, CopyOrder, StorageError, StorageResult};
use crate::tensor::ShardExtent;

/// One checked low read/write selection before placement. Byte traversals are
/// relative to `shard`; `backing` locates that origin within shared storage.
pub(crate) struct BoundView<'a> {
    pub shard: &'a BlockValue,
    pub extents: &'a [ShardExtent],
    pub backing: (BlockValueId, i64),
}

/// Physical requirements of a buffer access, independent of numerical format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StorageAccess {
    pub alignment: u32,
    pub access_tail_bytes: u32,
}
impl StorageAccess {
    pub(crate) fn include(&mut self, other: Self) {
        self.alignment = self.alignment.max(other.alignment);
        self.access_tail_bytes = self.access_tail_bytes.max(other.access_tail_bytes);
    }
}

/// Resolve a value's interpretation and backing origin before selecting either
/// tensor coordinates or raw byte accesses.
pub(crate) fn bind_storage(
    shards: &[BlockValue],
    id: BlockValueId,
) -> StorageResult<(&BlockValue, (BlockValueId, i64))> {
    let shard = shards
        .get(id.index() as usize)
        .ok_or(StorageError::WrongShard)?;
    if shard.id != id {
        return Err(StorageError::WrongShard);
    }
    let backing = storage_location(shards, id);
    let root = shards
        .get(backing.0.index() as usize)
        .ok_or(StorageError::WrongShard)?;
    if root.id != backing.0
        || root.tile != shard.tile
        || matches!(
            root.definition,
            ShardDefinition::Alias(_)
                | ShardDefinition::WritableAlias(_)
                | ShardDefinition::ShiftedAlias { .. }
        )
    {
        return Err(StorageError::InvalidView);
    }
    Ok((shard, backing))
}

impl ShardView {
    pub(crate) fn bind<'a>(&'a self, shards: &'a [BlockValue]) -> StorageResult<BoundView<'a>> {
        let (shard, backing) = bind_storage(shards, self.shard)?;
        crate::storage::validate_view(shard.storage(), &self.extents)?;
        Ok(BoundView {
            shard,
            extents: &self.extents,
            backing,
        })
    }
}

impl BoundView<'_> {
    pub(crate) fn geometry(
        &self,
        cache: &crate::storage::GeometryCache,
        order: CopyOrder,
    ) -> StorageResult<std::sync::Arc<crate::storage::GeometryView>> {
        cache.view(self.shard.storage(), self.extents, order)
    }

    pub(crate) fn traversal(&self, order: CopyOrder) -> StorageResult<ByteTraversal> {
        let storage = self.shard.storage();
        match order {
            CopyOrder::Panels => crate::storage::panel_byte_traversal(storage, self.extents),
            CopyOrder::Semantic | CopyOrder::Physical => {
                crate::storage::byte_traversal(storage, self.extents, order == CopyOrder::Physical)
            }
        }
    }
}

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

#[cfg(test)]
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

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum AddressError {
    #[error("shard {0} has no assigned address")]
    UnplacedShard(u32),
    #[error("storage address arithmetic overflowed")]
    Overflow,
}

/// Bind a value-relative byte offset to placement or the closest enclosing
/// Repeat override. Looking up only the ultimate root would lose overrides on
/// arguments which themselves alias another value.
pub(crate) fn resolve_address(
    shards: &[BlockValue],
    addresses: &std::collections::BTreeMap<BlockValueId, u32>,
    overrides: &std::collections::BTreeMap<BlockValueId, crate::TileAddress>,
    shard: BlockValueId,
    offset: u32,
) -> Result<crate::TileAddress, AddressError> {
    use crate::TileAddress;
    let overridden = if overrides.is_empty() {
        None
    } else {
        storage_chain(shards, shard).find_map(|(source, displacement)| {
            overrides.get(&source).map(|&base| (base, displacement))
        })
    };
    let (base, displacement) = match overridden {
        Some(binding) => binding,
        None => (
            TileAddress::Absolute(
                *addresses
                    .get(&shard)
                    .ok_or(AddressError::UnplacedShard(shard.index()))?,
            ),
            0,
        ),
    };
    let displaced = |base: i64| {
        base.checked_add(displacement)
            .and_then(|base| base.checked_add(i64::from(offset)))
            .ok_or(AddressError::Overflow)
    };
    Ok(match base {
        TileAddress::Absolute(base) => TileAddress::Absolute(
            u32::try_from(displaced(i64::from(base))?).map_err(|_| AddressError::Overflow)?,
        ),
        TileAddress::RepeatPointer { index, offset } => TileAddress::RepeatPointer {
            index,
            offset: i32::try_from(displaced(i64::from(offset))?)
                .map_err(|_| AddressError::Overflow)?,
        },
    })
}

pub(crate) fn storage_root(shards: &[BlockValue], shard: BlockValueId) -> BlockValueId {
    storage_location(shards, shard).0
}

/// Byte origin relative to the ultimate backing value, before placement.
pub(crate) fn storage_location(shards: &[BlockValue], shard: BlockValueId) -> (BlockValueId, i64) {
    storage_chain(shards, shard).last().unwrap()
}

/// Follow backing storage while retaining the byte displacement at each binding.
/// Repeat arguments may acquire other shard IDs through views and reductions.
pub(crate) fn storage_chain(
    shards: &[BlockValue],
    shard: BlockValueId,
) -> impl Iterator<Item = (BlockValueId, i64)> + '_ {
    std::iter::successors(Some((shard, 0i64)), |&(shard, offset)| {
        match shards.get(shard.index() as usize)?.definition {
            ShardDefinition::Alias(source) | ShardDefinition::WritableAlias(source) => {
                Some((source, offset))
            }
            ShardDefinition::ShiftedAlias {
                source,
                offset: delta,
            } => Some((source, offset + i64::from(delta))),
            _ => None,
        }
    })
    .take(shards.len().saturating_add(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, Precision, TensorType};

    #[test]
    fn alias_binding_preserves_the_selected_type_strides_and_signed_origin() {
        let storage = BlockValue {
            id: BlockValueId(0),
            tile: 3,
            tensor_type: TensorType::new([4, 16], Precision::F16, Layout::row_sharded(1)),
            extents: [4, 16]
                .into_iter()
                .enumerate()
                .map(|(axis, end)| ShardExtent {
                    axis: axis as u16,
                    start: 0,
                    logical_end: end,
                    physical_end: end,
                })
                .collect(),
            definition: ShardDefinition::Staging,
        };
        let mut shards = vec![storage];
        for definition in [
            ShardDefinition::ShiftedAlias {
                source: BlockValueId(0),
                offset: -32,
            },
            ShardDefinition::Alias(BlockValueId(1)),
            ShardDefinition::ShiftedAlias {
                source: BlockValueId(2),
                offset: 16,
            },
        ] {
            let mut alias = shards.last().unwrap().clone();
            alias.id = BlockValueId(shards.len() as u32);
            alias.tensor_type.format.precision = Precision::F8F143 { scale_exponent: -4 };
            alias.definition = definition;
            shards.push(alias);
        }
        let view = ShardView {
            shard: BlockValueId(3),
            extents: vec![
                ShardExtent {
                    axis: 0,
                    start: 1,
                    logical_end: 3,
                    physical_end: 3,
                },
                ShardExtent {
                    axis: 1,
                    start: 2,
                    logical_end: 6,
                    physical_end: 6,
                },
            ],
        };
        let bound = view.bind(&shards).unwrap();
        let addresses = bound
            .traversal(CopyOrder::Semantic)
            .unwrap()
            .spans()
            .flat_map(|span| {
                (span.offset..span.offset + span.bytes)
                    .map(|byte| bound.backing.1 + i64::from(byte))
            })
            .collect::<Vec<_>>();
        assert_eq!(bound.backing.0, BlockValueId(0));
        assert_eq!(addresses, vec![2, 3, 4, 5, 18, 19, 20, 21]);
        assert_eq!(bound.shard.tensor_type.format.precision.bytes(), 1);

        shards[0].definition = ShardDefinition::Alias(BlockValueId(3));
        assert!(view.bind(&shards).is_err(), "alias cycle");
        shards[0].definition = ShardDefinition::Alias(BlockValueId(9));
        assert!(view.bind(&shards).is_err(), "missing backing");
    }
}
