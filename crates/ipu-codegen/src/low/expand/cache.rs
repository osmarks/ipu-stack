//! Per-build caches for relative copy descriptors and destination geometry.
//! Packing selection and low graph mutation remain in movement construction.
use super::ExpansionResult;
use crate::low::{BlockValue, BlockValueId, CopyOperation, ShardView};
use crate::storage::{CopyGeometry, CopyMapping, CopyOrder, ViewGeometry};
use crate::tensor::{ElementOrder, Precision, ShardExtent};
use hashbrown::HashTable;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;
use std::sync::Mutex;

const MAX_ENTRIES: usize = 32768;

// Prehashed lookup compares borrowed views; hit paths do not clone the large
// destination keys. HashTable owns probing/collision handling and reuses the
// recorded hash when growing, avoiding a second hash or a vector per bucket.
struct Memo<K, V> {
    entries: Mutex<Entries<K, V>>,
}
struct Entries<K, V> {
    table: HashTable<(u64, K, Arc<V>)>,
    hits: u64,
    misses: u64,
}
impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(Entries {
                table: HashTable::new(),
                hits: 0,
                misses: 0,
            }),
        }
    }
}
impl<K: Eq, V> Memo<K, V> {
    fn get(&self, hash: u64, matches: impl Fn(&K) -> bool) -> Option<Arc<V>> {
        let mut state = self.entries.lock().unwrap();
        let found = state
            .table
            .find(hash, |(_, key, _)| matches(key))
            .map(|(_, _, value)| Arc::clone(value));
        if found.is_some() {
            state.hits += 1;
        } else {
            state.misses += 1;
        }
        found
    }
    fn insert(&self, hash: u64, key: K, value: Arc<V>, limit: usize) {
        let mut state = self.entries.lock().unwrap();
        if state.table.len() < limit
            && state
                .table
                .find(hash, |(_, existing, _)| *existing == key)
                .is_none()
        {
            state
                .table
                .insert_unique(hash, (hash, key, value), |entry| entry.0);
        }
    }

    fn has_capacity(&self, limit: usize) -> bool {
        self.entries.lock().unwrap().table.len() < limit
    }
    fn stats(&self) -> (usize, u64, u64) {
        let state = self.entries.lock().unwrap();
        (state.table.len(), state.hits, state.misses)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct CopyKey {
    source: ViewGeometry,
    destination: ViewGeometry,
    order: CopyOrder,
    same_buffer: bool,
}

#[derive(PartialEq, Eq, Hash)]
struct GeometrySource {
    format: (Precision, ElementOrder),
    allocation: Vec<ShardExtent>,
    source: Vec<ShardExtent>,
    destination: Vec<ShardExtent>,
}
#[derive(PartialEq, Eq, Hash)]
struct GeometryKey {
    destination: (Precision, ElementOrder),
    allocation: Vec<ShardExtent>,
    mappings: Vec<GeometrySource>,
    order: CopyOrder,
}

// Relative traversal depends on byte interpretation. Shape, tile ownership,
// replicas and bank class do not affect it and must not split cache entries.
fn format_key(shard: &BlockValue) -> (Precision, ElementOrder) {
    (
        shard.tensor_type.format.precision,
        shard.tensor_type.format.layout.order,
    )
}

pub(crate) struct ExpansionCache {
    enabled: bool,
    copies: Memo<CopyKey, Vec<CopyOperation<()>>>,
    geometry: Memo<GeometryKey, CopyGeometry>,
}
impl Default for ExpansionCache {
    fn default() -> Self {
        Self {
            enabled: true,
            copies: Memo::default(),
            geometry: Memo::default(),
        }
    }
}
impl ExpansionCache {
    pub(crate) fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub(crate) fn stats(&self) -> (usize, u64, u64) {
        self.copies.stats()
    }
    pub(crate) fn geometry_stats(&self) -> (usize, u64, u64) {
        self.geometry.stats()
    }

    pub(super) fn geometry(
        &self,
        shards: &[BlockValue],
        mappings: &[(ShardView, ShardView)],
        destination: BlockValueId,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<CopyGeometry>> {
        let shard = &shards[destination.index() as usize];
        for (source, target) in mappings {
            source.bind(shards)?;
            target.bind(shards)?;
            if target.shard != destination {
                return Err(super::ExpansionError::InvalidCopyPlan);
            }
        }
        let generate = || {
            let mappings = mappings
                .iter()
                .map(|(source, destination)| CopyMapping {
                    source: shards[source.shard.index() as usize].storage(),
                    source_extents: &source.extents,
                    destination_extents: &destination.extents,
                })
                .collect::<Vec<_>>();
            CopyGeometry::analyze(
                shard.storage(),
                &mappings,
                order,
                ipu_exchange::MAX_TRANSFER_WORDS * 4,
            )
        };
        if !self.enabled
            || (order != CopyOrder::Semantic
                && mappings.len() == 1
                && mappings[0].1.extents == shard.extents)
        {
            return Ok(Arc::new(generate()?));
        }
        let mut hash = foldhash::fast::FixedState::default().build_hasher();
        format_key(shard).hash(&mut hash);
        shard.extents.hash(&mut hash);
        order.hash(&mut hash);
        mappings.len().hash(&mut hash);
        for (source, destination) in mappings {
            let input = &shards[source.shard.index() as usize];
            format_key(input).hash(&mut hash);
            input.extents.hash(&mut hash);
            source.extents.hash(&mut hash);
            destination.extents.hash(&mut hash);
        }
        let hash = hash.finish();
        if let Some(geometry) = self.geometry.get(hash, |key| {
            key.destination == format_key(shard)
                && key.allocation == shard.extents
                && key.order == order
                && key.mappings.len() == mappings.len()
                && key
                    .mappings
                    .iter()
                    .zip(mappings)
                    .all(|(key, (source, destination))| {
                        let input = &shards[source.shard.index() as usize];
                        key.format == format_key(input)
                            && key.allocation == input.extents
                            && key.source == source.extents
                            && key.destination == destination.extents
                    })
        }) {
            return Ok(geometry);
        }
        if !self.geometry.has_capacity(MAX_ENTRIES) {
            return Ok(Arc::new(generate()?));
        }
        let key = GeometryKey {
            destination: format_key(shard),
            allocation: shard.extents.clone(),
            order,
            mappings: mappings
                .iter()
                .map(|(source, destination)| {
                    let shard = &shards[source.shard.index() as usize];
                    GeometrySource {
                        format: format_key(shard),
                        allocation: shard.extents.clone(),
                        source: source.extents.clone(),
                        destination: destination.extents.clone(),
                    }
                })
                .collect(),
        };
        let geometry = Arc::new(generate()?);
        self.geometry
            .insert(hash, key, Arc::clone(&geometry), MAX_ENTRIES);
        Ok(geometry)
    }

    pub(super) fn copy(
        &self,
        shards: &[BlockValue],
        source: &ShardView,
        destination: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<Vec<CopyOperation<()>>>> {
        let source = source.bind(shards)?;
        let destination = destination.bind(shards)?;
        let generate = || -> ExpansionResult<Vec<CopyOperation<()>>> {
            let a = source.traversal(order)?;
            let b = destination.traversal(order)?;
            // Alias IDs can differ while their storage overlaps. Reordering
            // copy spans is only safe between different backing allocations.
            let copies =
                CopyOperation::from_traversals(source.backing.0, destination.backing.0, &a, &b)?
                    .into_iter()
                    .map(|c| CopyOperation {
                        source: (),
                        destination: (),
                        source_offset: c.source_offset,
                        destination_offset: c.destination_offset,
                        bytes: c.bytes,
                        pattern: c.pattern,
                    })
                    .collect();
            Ok(copies)
        };
        let left = source.shard;
        let right = destination.shard;
        let whole_copy = source.extents == left.extents
            && destination.extents == right.extents
            && (order == CopyOrder::Physical
                || (order == CopyOrder::Semantic
                    && left.tensor_type.format.layout.order == ElementOrder::RowMajor
                    && right.tensor_type.format.layout.order == ElementOrder::RowMajor));
        if !self.enabled || whole_copy {
            return Ok(Arc::new(generate()?));
        }
        let key = CopyKey {
            source: ViewGeometry::new(left.storage(), source.extents)?,
            destination: ViewGeometry::new(right.storage(), destination.extents)?,
            order,
            same_buffer: source.backing.0 == destination.backing.0,
        };
        let mut hash = foldhash::fast::FixedState::default().build_hasher();
        key.hash(&mut hash);
        let hash = hash.finish();
        if let Some(copies) = self.copies.get(hash, |existing| *existing == key) {
            return Ok(copies);
        }
        let copies = generate()?;
        // Generation is outside the lock, so unrelated finalists do not serialize.
        let result = Arc::new(copies);
        self.copies
            .insert(hash, key, Arc::clone(&result), MAX_ENTRIES);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::low::ShardDefinition;
    use crate::tensor::{Layout, TensorType};

    #[test]
    fn memo_checks_full_keys_and_bounds_entries_even_on_hash_collisions() {
        let memo = Memo::default();
        memo.insert(7, 1, Arc::new(10), 2);
        memo.insert(7, 2, Arc::new(20), 2);
        memo.insert(7, 3, Arc::new(30), 2);
        assert_eq!(*memo.get(7, |key| *key == 1).unwrap(), 10);
        assert_eq!(*memo.get(7, |key| *key == 2).unwrap(), 20);
        assert!(memo.get(7, |key| *key == 3).is_none());
        assert_eq!(memo.stats(), (2, 2, 1));

        // Growth must retain the supplied fingerprints, even though they need
        // not equal a hash recomputed from the owned key.
        let memo = Memo::default();
        for key in 0..1024 {
            memo.insert(key % 7, key, Arc::new(key * 3), 1024);
        }
        for key in 0..1024 {
            assert_eq!(
                *memo.get(key % 7, |stored| *stored == key).unwrap(),
                key * 3
            );
        }
    }

    #[test]
    fn copy_fragments_share_relative_geometry_across_tiles_and_origins() {
        let cache = ExpansionCache::default();
        let mut first = None;
        for shift in [0, 16] {
            let shards = (0..2)
                .map(|id| BlockValue {
                    id: BlockValueId(id),
                    tile: id as u16,
                    tensor_type: TensorType::new([2, 64], Precision::F16, Layout::row_sharded(1)),
                    extents: vec![
                        ShardExtent {
                            axis: 0,
                            start: 0,
                            logical_end: 2,
                            physical_end: 2,
                        },
                        ShardExtent {
                            axis: 1,
                            start: shift,
                            logical_end: shift + 8,
                            physical_end: shift + 8,
                        },
                    ],
                    definition: ShardDefinition::Staging,
                })
                .collect::<Vec<_>>();
            let view = |id: usize| {
                let mut extents = shards[id].extents.clone();
                extents[1].start += 2;
                extents[1].logical_end -= 2;
                extents[1].physical_end -= 2;
                ShardView {
                    shard: BlockValueId(id as u32),
                    extents,
                }
            };
            let result = cache
                .copy(&shards, &view(0), &view(1), CopyOrder::Semantic)
                .unwrap();
            if let Some(first) = &first {
                assert!(Arc::ptr_eq(first, &result));
            } else {
                first = Some(result);
            }
        }
        assert_eq!(cache.stats(), (1, 1, 1));
    }

    #[test]
    fn destination_geometry_ignores_ownership_but_tracks_byte_interpretation() {
        let mut shards = (0..2)
            .map(|id| BlockValue {
                id: BlockValueId(id),
                tile: id as u16,
                tensor_type: TensorType::new([2, 64], Precision::F16, Layout::row_sharded(1)),
                extents: vec![
                    ShardExtent {
                        axis: 0,
                        start: 0,
                        logical_end: 2,
                        physical_end: 2,
                    },
                    ShardExtent {
                        axis: 1,
                        start: 0,
                        logical_end: 64,
                        physical_end: 64,
                    },
                ],
                definition: ShardDefinition::Staging,
            })
            .collect::<Vec<_>>();
        let views = shards
            .iter()
            .map(|shard| ShardView {
                shard: shard.id,
                extents: shard.extents.clone(),
            })
            .collect::<Vec<_>>();
        let mappings = [(views[0].clone(), views[1].clone())];
        let cache = ExpansionCache::default();
        let before = cache
            .geometry(&shards, &mappings, BlockValueId(1), CopyOrder::Semantic)
            .unwrap();
        for shard in &mut shards {
            shard.tile += 2;
            shard.tensor_type.shape.0[0] *= 2;
            shard.tensor_type.format.layout = Layout::row_sharded(2);
            shard.tensor_type.format.layout.memory_class = crate::MemoryClass::Ipu21Interleaved;
        }
        let remapped = cache
            .geometry(&shards, &mappings, BlockValueId(1), CopyOrder::Semantic)
            .unwrap();
        assert!(Arc::ptr_eq(&before, &remapped));
        for shard in &mut shards {
            shard.tensor_type.format.precision = Precision::F32;
        }
        let wider = cache
            .geometry(&shards, &mappings, BlockValueId(1), CopyOrder::Semantic)
            .unwrap();
        assert!(!Arc::ptr_eq(&before, &wider));
        assert_eq!(wider.bytes, before.bytes * 2);
        assert_eq!(cache.geometry_stats(), (2, 1, 2));
    }
    #[test]
    fn copy_preparation_preserves_order_when_distinct_values_share_storage() {
        fn mapping(copies: &[CopyOperation<()>]) -> Vec<(u32, u32)> {
            copies
                .iter()
                .flat_map(|copy| {
                    let (rows, bytes, source_stride, destination_stride) = match copy.pattern {
                        crate::CopyPattern::Contiguous => (1, copy.bytes, 0, 0),
                        crate::CopyPattern::Strided {
                            rows,
                            row_bytes,
                            source_stride,
                            destination_stride,
                        } => (rows, row_bytes, source_stride, destination_stride),
                    };
                    (0..rows).flat_map(move |row| {
                        (0..bytes).map(move |byte| {
                            (
                                copy.source_offset + row * source_stride + byte,
                                copy.destination_offset + row * destination_stride + byte,
                            )
                        })
                    })
                })
                .collect()
        }
        let orders = [
            ElementOrder::RowMajor,
            ElementOrder::Amp(crate::AmpOrder::Left),
            ElementOrder::Amp(crate::AmpOrder::Output),
            ElementOrder::Amp(crate::AmpOrder::TransposedLeft),
            ElementOrder::Amp(crate::AmpOrder::TransposedOutput),
            ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                row_block: 16,
                column_block: 32,
            }),
        ];
        let mut reordered = 0;
        for precision in [Precision::F16, Precision::F32] {
            for source_order in orders {
                for destination_order in orders {
                    let mut shards = [source_order, destination_order]
                        .into_iter()
                        .enumerate()
                        .map(|(index, order)| {
                            let mut layout = Layout::row_sharded(1);
                            layout.order = order;
                            BlockValue {
                                id: BlockValueId(index as u32),
                                tile: 0,
                                tensor_type: TensorType::new([48, 96], precision, layout),
                                extents: [48, 96]
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
                            }
                        })
                        .collect::<Vec<_>>();
                    let view = |id| ShardView {
                        shard: BlockValueId(id),
                        extents: vec![
                            ShardExtent {
                                axis: 0,
                                start: 0,
                                logical_end: 32,
                                physical_end: 32,
                            },
                            ShardExtent {
                                axis: 1,
                                start: 0,
                                logical_end: 64,
                                physical_end: 64,
                            },
                        ],
                    };
                    let (source, destination) = (view(0), view(1));
                    let a = source
                        .bind(&shards)
                        .unwrap()
                        .traversal(CopyOrder::Semantic)
                        .unwrap();
                    let b = destination
                        .bind(&shards)
                        .unwrap()
                        .traversal(CopyOrder::Semantic)
                        .unwrap();
                    let mut expected = Vec::new();
                    crate::storage::for_each_copy_span(a.spans(), b.spans(), |a, b, bytes| {
                        expected.extend((0..bytes).map(|byte| (a + byte, b + byte)));
                        Ok(())
                    })
                    .unwrap();
                    let cache = ExpansionCache::default();
                    let separate = cache
                        .copy(&shards, &source, &destination, CopyOrder::Semantic)
                        .unwrap();
                    reordered += usize::from(mapping(&separate) != expected);
                    for offset in [0, 64] {
                        shards[1].definition = ShardDefinition::ShiftedAlias {
                            source: BlockValueId(0),
                            offset,
                        };
                        let shared = cache
                            .copy(&shards, &source, &destination, CopyOrder::Semantic)
                            .unwrap();
                        assert_eq!(mapping(&shared), expected);
                    }
                }
            }
        }
        assert!(
            reordered > 0,
            "fixture must expose unsafe reuse of a reordered disjoint copy"
        );
    }
}
