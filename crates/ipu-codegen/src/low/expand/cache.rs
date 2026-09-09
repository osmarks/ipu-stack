//! Per-build caches for relative copy fragments and copy preparation plans.
//! Phase assembly, alias
//! mutation and deferred materialization stay in the caller.
use super::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

const MAX_ENTRIES: usize = 32768;

struct Memo<K, V> {
    entries: Mutex<(HashMap<u64, Vec<(K, Arc<V>)>>, u64, u64, usize)>,
}
impl<K, V> Default for Memo<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new((HashMap::new(), 0, 0, 0)),
        }
    }
}
impl<K: Eq, V> Memo<K, V> {
    fn get(&self, hash: u64, matches: impl Fn(&K) -> bool) -> Option<Arc<V>> {
        let mut state = self.entries.lock().unwrap();
        let found = state
            .0
            .get(&hash)
            .and_then(|bucket| bucket.iter().find(|(key, _)| matches(key)))
            .map(|(_, value)| Arc::clone(value));
        if found.is_some() {
            state.1 += 1;
        } else {
            state.2 += 1;
        }
        found
    }
    fn insert(&self, hash: u64, key: K, value: Arc<V>, limit: usize) {
        let mut state = self.entries.lock().unwrap();
        if state.3 < limit {
            let bucket = state.0.entry(hash).or_default();
            if !bucket.iter().any(|(existing, _)| *existing == key) {
                bucket.push((key, value));
                state.3 += 1;
            }
        }
    }
    fn has_capacity(&self, limit: usize) -> bool {
        self.entries.lock().unwrap().3 < limit
    }
    fn stats(&self) -> (usize, u64, u64) {
        let state = self.entries.lock().unwrap();
        (state.3, state.1, state.2)
    }
}

#[derive(PartialEq, Eq, Hash)]
struct CopyKey {
    source: crate::storage::ViewGeometry,
    destination: crate::storage::ViewGeometry,
    order: CopyOrder,
    same_buffer: bool,
}

#[derive(PartialEq, Eq, Hash)]
struct PlanSource {
    format: TensorFormat,
    allocation: Vec<ShardExtent>,
    source: Vec<ShardExtent>,
    destination: Vec<ShardExtent>,
}
#[derive(PartialEq, Eq, Hash)]
struct PlanKey {
    destination: TensorType,
    allocation: Vec<ShardExtent>,
    mappings: Vec<PlanSource>,
    order: CopyOrder,
}

pub(crate) struct ExpansionCache {
    enabled: bool,
    copies: Memo<CopyKey, Vec<CopyOperation<()>>>,
    plans: Memo<PlanKey, crate::CopyPlan>,
}
impl Default for ExpansionCache {
    fn default() -> Self {
        Self {
            enabled: true,
            copies: Memo::default(),
            plans: Memo::default(),
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
    pub(crate) fn plan_stats(&self) -> (usize, u64, u64) {
        self.plans.stats()
    }

    pub(super) fn plan(
        &self,
        shards: &[BlockValue],
        mappings: &[(ShardView, ShardView)],
        destination: BlockValueId,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<crate::CopyPlan>> {
        let shard = &shards[destination.index() as usize];
        let generate = || {
            let mappings = mappings
                .iter()
                .map(|(source, destination)| crate::CopyMapping {
                    source: shards[source.shard.index() as usize].storage(),
                    source_extents: &source.extents,
                    destination_extents: &destination.extents,
                })
                .collect::<Vec<_>>();
            crate::CopyPlan::for_destination(&shard.tensor_type, &shard.extents, &mappings, order)
        };
        if !self.enabled
            || (order != CopyOrder::Semantic
                && mappings.len() == 1
                && mappings[0].1.extents == shard.extents)
        {
            return Ok(Arc::new(generate()?));
        }
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        shard.tensor_type.hash(&mut hash);
        shard.extents.hash(&mut hash);
        order.hash(&mut hash);
        mappings.len().hash(&mut hash);
        for (source, destination) in mappings {
            let input = &shards[source.shard.index() as usize];
            input.tensor_type.format.hash(&mut hash);
            input.extents.hash(&mut hash);
            source.extents.hash(&mut hash);
            destination.extents.hash(&mut hash);
        }
        let hash = hash.finish();
        if let Some(plan) = self.plans.get(hash, |key| {
            key.destination == shard.tensor_type
                && key.allocation == shard.extents
                && key.order == order
                && key.mappings.len() == mappings.len()
                && key
                    .mappings
                    .iter()
                    .zip(mappings)
                    .all(|(key, (source, destination))| {
                        let input = &shards[source.shard.index() as usize];
                        key.format == input.tensor_type.format
                            && key.allocation == input.extents
                            && key.source == source.extents
                            && key.destination == destination.extents
                    })
        }) {
            return Ok(plan);
        }
        if !self.plans.has_capacity(MAX_ENTRIES) {
            return Ok(Arc::new(generate()?));
        }
        let key = PlanKey {
            destination: shard.tensor_type.clone(),
            allocation: shard.extents.clone(),
            order,
            mappings: mappings
                .iter()
                .map(|(source, destination)| {
                    let shard = &shards[source.shard.index() as usize];
                    PlanSource {
                        format: shard.tensor_type.format.clone(),
                        allocation: shard.extents.clone(),
                        source: source.extents.clone(),
                        destination: destination.extents.clone(),
                    }
                })
                .collect(),
        };
        let plan = Arc::new(generate()?);
        self.plans.insert(hash, key, Arc::clone(&plan), MAX_ENTRIES);
        Ok(plan)
    }

    pub(super) fn copy(
        &self,
        shards: &[BlockValue],
        source: &ShardView,
        destination: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<Vec<CopyOperation<()>>>> {
        let generate = || -> ExpansionResult<Vec<CopyOperation<()>>> {
            let a = view_byte_traversal(&shards[source.shard.index() as usize], source, order)?;
            let b = view_byte_traversal(
                &shards[destination.shard.index() as usize],
                destination,
                order,
            )?;
            let copies = LocalCopy::from_traversals(source.shard, destination.shard, &a, &b)?
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
        let left = &shards[source.shard.index() as usize];
        let right = &shards[destination.shard.index() as usize];
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
            source: crate::storage::ViewGeometry::new(left.storage(), &source.extents)?,
            destination: crate::storage::ViewGeometry::new(right.storage(), &destination.extents)?,
            order,
            same_buffer: source.shard == destination.shard,
        };
        let mut hash = std::collections::hash_map::DefaultHasher::new();
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
}
