//! Shared allocation-relative geometry for movement construction and costing.
//! Keys contain byte interpretation and normalized selections, never low value
//! IDs, placement, kernel choices or exchange timing policy. Interned IDs are
//! unique across caches and let compound keys share their view descriptions.

use super::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

type Map<K, V> = HashMap<K, V, foldhash::fast::FixedState>;
static NEXT_VIEW: AtomicU64 = AtomicU64::new(0);

struct Memo<K, V> {
    limit: usize,
    state: Mutex<Map<K, Arc<V>>>,
}
impl<K: Eq + Hash, V> Memo<K, V> {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            state: Mutex::new(Map::default()),
        }
    }
    fn get(&self, key: K, build: impl FnOnce() -> StorageResult<V>) -> StorageResult<Arc<V>> {
        {
            let state = self.state.lock().unwrap();
            if let Some(value) = state.get(&key).cloned() {
                return Ok(value);
            }
        }
        // Independent candidates may construct unrelated facts concurrently.
        let value = Arc::new(build()?);
        let mut state = self.state.lock().unwrap();
        if state.len() < self.limit {
            return Ok(Arc::clone(state.entry(key).or_insert(value)));
        }
        Ok(value)
    }
}

pub(crate) struct GeometryView {
    id: u64,
    pub traversal: ByteTraversal,
}

/// Canonical matched stream pieces. Regular pairs retain affine rows; irregular
/// pairs retain their exact stream boundaries. No launch coalescing or reorder
/// is implied by this representation.
pub(crate) struct CopyPair {
    pub bytes: u64,
    pub rows: Vec<[StridedSpan; 2]>,
}
impl CopyPair {
    pub(crate) fn new(source: &ByteTraversal, target: &ByteTraversal) -> StorageResult<Self> {
        if source.byte_len() != target.byte_len() {
            return Err(StorageError::InvalidView);
        }
        if source.byte_len() == 0 {
            return Ok(Self {
                bytes: 0,
                rows: Vec::new(),
            });
        }
        if let Some((a, b)) = source.regular_copy(target) {
            return Ok(Self {
                bytes: source.byte_len(),
                rows: vec![[a, b]],
            });
        }
        Self::from_spans(source.spans(), target.spans())
    }
    pub(crate) fn from_spans(
        source: impl IntoIterator<Item = ByteSpan>,
        target: impl IntoIterator<Item = ByteSpan>,
    ) -> StorageResult<Self> {
        let mut result = Self {
            bytes: 0,
            rows: Vec::new(),
        };
        for_each_copy_span(source, target, |source, target, bytes| {
            result.bytes += u64::from(bytes);
            result.rows.push([source, target].map(|offset| StridedSpan {
                offset,
                bytes,
                rows: 1,
                stride: bytes,
            }));
            Ok(())
        })?;
        Ok(result)
    }
    pub(crate) fn fragments(&self, maximum_bytes: u32) -> u64 {
        self.rows
            .iter()
            .map(|[source, _]| {
                u64::from(source.rows) * u64::from(source.bytes.div_ceil(maximum_bytes))
            })
            .sum()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CopyMapping<'a> {
    pub source: TensorStorage<'a>,
    pub source_extents: &'a [ShardExtent],
    pub destination_extents: &'a [ShardExtent],
}

#[derive(PartialEq, Eq, Hash)]
struct DestinationKey {
    allocation: u64,
    coverage: Vec<u64>,
    pairs: Option<Vec<(u64, u64)>>,
    same_element_order: bool,
    maximum_fragment_bytes: u32,
}

/// Exact population facts. Coverage stays symbolic until a selected realization
/// needs its holes; packing kernels may already write the padding themselves.
pub(crate) struct DestinationGeometry {
    pub bytes: u32,
    coverage: ByteTraversal,
    uncovered: OnceLock<StorageResult<Vec<ByteSpan>>>,
    pub fragments: Option<u64>,
    pub semantic: bool,
    pub destination_word_aligned: bool,
    pub same_element_order: bool,
}
impl DestinationGeometry {
    pub(crate) fn uncovered(&self) -> StorageResult<&[ByteSpan]> {
        self.uncovered
            .get_or_init(|| super::movement::uncovered_ranges(self.bytes, &self.coverage))
            .as_ref()
            .map(Vec::as_slice)
            .map_err(Clone::clone)
    }
}

pub(crate) struct GeometryCache {
    views: Memo<(CopyOrder, ViewGeometry), GeometryView>,
    pairs: Memo<(u64, u64), CopyPair>,
    destinations: Memo<DestinationKey, DestinationGeometry>,
}
impl Default for GeometryCache {
    fn default() -> Self {
        Self {
            // The measured MLP B2 and SigLIP candidates contain up to 60k views
            // and 150k pairs. These are entry limits, not byte budgets.
            views: Memo::new(131_072),
            pairs: Memo::new(262_144),
            destinations: Memo::new(32_768),
        }
    }
}
impl GeometryCache {
    pub(crate) fn view(
        &self,
        storage: TensorStorage<'_>,
        extents: &[ShardExtent],
        order: CopyOrder,
    ) -> StorageResult<Arc<GeometryView>> {
        let key = (order, ViewGeometry::new(storage, extents)?);
        self.views.get(key, || {
            Ok(GeometryView {
                id: NEXT_VIEW.fetch_add(1, Ordering::Relaxed),
                traversal: match order {
                    CopyOrder::Panels => panel_byte_traversal(storage, extents)?,
                    _ => byte_traversal(storage, extents, order == CopyOrder::Physical)?,
                },
            })
        })
    }
    pub(crate) fn pair(
        &self,
        source: &GeometryView,
        target: &GeometryView,
    ) -> StorageResult<Arc<CopyPair>> {
        self.pairs.get((source.id, target.id), || {
            CopyPair::new(&source.traversal, &target.traversal)
        })
    }
    pub(crate) fn destination(
        &self,
        destination: TensorStorage<'_>,
        mappings: &[CopyMapping<'_>],
        order: CopyOrder,
        maximum_fragment_bytes: u32,
    ) -> StorageResult<Arc<DestinationGeometry>> {
        if maximum_fragment_bytes == 0 {
            return Err(StorageError::InvalidView);
        }
        let allocation = self.view(destination, destination.extents, CopyOrder::Physical)?;
        // Physical/panel population needs only destination coverage. Inspect
        // source traversal only for semantic copies, where fragmentation matters.
        let views = if order == CopyOrder::Semantic {
            mappings
                .iter()
                .map(|m| {
                    Ok((
                        self.view(m.source, m.source_extents, order)?,
                        self.view(destination, m.destination_extents, order)?,
                    ))
                })
                .collect::<StorageResult<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let covered = mappings
            .iter()
            .map(|mapping| {
                let mut extents = mapping.destination_extents.to_vec();
                if order == CopyOrder::Semantic {
                    for extent in &mut extents {
                        extent.physical_end = extent.logical_end;
                    }
                }
                self.view(destination, &extents, CopyOrder::Physical)
            })
            .collect::<StorageResult<Vec<_>>>()?;
        let same_element_order = mappings
            .iter()
            .all(|m| m.source.format.layout.order == destination.format.layout.order);
        let key = DestinationKey {
            allocation: allocation.id,
            coverage: covered.iter().map(|view| view.id).collect(),
            pairs: (order == CopyOrder::Semantic).then(|| {
                views
                    .iter()
                    .map(|(source, target)| (source.id, target.id))
                    .collect()
            }),
            same_element_order,
            maximum_fragment_bytes,
        };
        self.destinations.get(key, || {
            let mut fragments = (order == CopyOrder::Semantic).then_some(0u64);
            let mut destination_unaligned = false;
            for (source, target) in &views {
                destination_unaligned |= !target.traversal.word_aligned();
                fragments = match fragments {
                    Some(count) if source.traversal.word_aligned() && !destination_unaligned => {
                        self.pair(source, target).ok().map(|pair| {
                            count.saturating_add(pair.fragments(maximum_fragment_bytes))
                        })
                    }
                    _ => None,
                };
            }
            Ok(DestinationGeometry {
                bytes: storage_bytes(destination)?,
                coverage: ByteTraversal::physical_union(
                    covered.iter().map(|v| v.traversal.clone()),
                ),
                uncovered: OnceLock::new(),
                fragments,
                semantic: order == CopyOrder::Semantic,
                destination_word_aligned: !destination_unaligned,
                same_element_order,
            })
        })
    }
}
