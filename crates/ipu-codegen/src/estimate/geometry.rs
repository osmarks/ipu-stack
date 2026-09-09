//! Geometry facts shared by execution and table-storage analysis. Tile bindings,
//! transmit lanes and pointer continuation remain in the consuming phase walk.
use crate::storage::{ByteTraversal, ViewGeometry};
use crate::{CopyOrder, ExpansionResult, ShardView, TileGraph};
use std::collections::HashMap;
use std::sync::Arc;

const LIMIT: u32 = ipu_exchange::MAX_TRANSFER_WORDS * 4;

#[derive(Clone, Copy)]
pub(super) struct ReceiveRows {
    pub offset: u32,
    pub bytes: u32,
    pub rows: u32,
    pub stride: u32,
}
pub(super) struct CopyGeometry {
    pub bytes: u64,
    pub fragments: u64,
    pub long_fragments: u64,
    pub receives: Vec<ReceiveRows>,
}
impl CopyGeometry {
    fn new(source: &ByteTraversal, target: &ByteTraversal) -> ExpansionResult<Self> {
        let mut result = Self {
            bytes: source.byte_len(),
            fragments: 0,
            long_fragments: 0,
            receives: Vec::new(),
        };
        if result.bytes != target.byte_len() {
            return Err(crate::storage::StorageError::InvalidView.into());
        }
        let mut append = |offset, bytes: u32, rows: u32, stride| {
            if bytes == 0 {
                return;
            }
            result.fragments += u64::from(rows) * u64::from(bytes.div_ceil(LIMIT));
            result.long_fragments += u64::from(rows)
                * (u64::from(bytes / LIMIT) * u64::from(LIMIT > 256)
                    + u64::from(bytes % LIMIT > 256));
            result.receives.push(ReceiveRows {
                offset,
                bytes,
                rows,
                stride,
            });
        };
        if let Some((_, target)) = source.regular_copy(target) {
            append(target.offset, target.bytes, target.rows, target.stride);
        } else {
            crate::for_each_copy_span(source.spans(), target.spans(), |_, offset, bytes| {
                append(offset, bytes, 1, bytes);
                Ok(())
            })?;
        }
        Ok(result)
    }
}

/// Scoped to one immutable expanded program. Integer geometry IDs make pair
/// lookup cheap and avoid retaining repeated traversal trees per recipient.
#[derive(Default)]
pub(crate) struct GeometryAnalysis {
    views: HashMap<(CopyOrder, ViewGeometry), usize>,
    traversals: Vec<ByteTraversal>,
    pairs: HashMap<(usize, usize), Arc<CopyGeometry>>,
}
impl GeometryAnalysis {
    pub(crate) fn stats(&self) -> (usize, usize, usize) {
        (
            self.traversals.len(),
            self.pairs.len(),
            self.pairs.values().map(|p| p.receives.len()).sum(),
        )
    }

    pub(super) fn bytes(&self, view: usize) -> u64 {
        self.traversals[view].byte_len()
    }
    pub(super) fn view(
        &mut self,
        program: &TileGraph,
        view: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<usize> {
        let shard = &program.shards[view.shard.index() as usize];
        let key = (order, ViewGeometry::new(shard.storage(), &view.extents)?);
        if let Some(&id) = self.views.get(&key) {
            return Ok(id);
        }
        let traversal = crate::view_byte_traversal(shard, view, order)?;
        let id = self.traversals.len();
        self.traversals.push(traversal);
        self.views.insert(key, id);
        Ok(id)
    }
    pub(super) fn copy(
        &mut self,
        program: &TileGraph,
        source: usize,
        target: &ShardView,
        order: CopyOrder,
    ) -> ExpansionResult<Arc<CopyGeometry>> {
        let a = source;
        let b = self.view(program, target, order)?;
        if let Some(pair) = self.pairs.get(&(a, b)) {
            return Ok(Arc::clone(pair));
        }
        let pair = Arc::new(CopyGeometry::new(&self.traversals[a], &self.traversals[b])?);
        self.pairs.insert((a, b), Arc::clone(&pair));
        Ok(pair)
    }
}
