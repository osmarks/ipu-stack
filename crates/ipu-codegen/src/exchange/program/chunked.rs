//! Shared chunks for speculative timelines and encoded prefixes.
use std::borrow::Cow;
use std::ops::{Index, Range};
use std::sync::Arc;

// Trials share immutable chunks and copy at most the final partial chunk.
// A flat chunk index keeps lookup/rollback bounded and avoids recursive drops.
const CHUNK_ITEMS: usize = 128;

#[derive(Clone, Debug)]
pub(super) struct Chunked<T> {
    chunks: Vec<Arc<Vec<T>>>,
    len: usize,
}

impl<T> Default for Chunked<T> {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }
}

impl<T: Clone> Chunked<T> {
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn get(&self, index: usize) -> Option<&T> {
        (index < self.len).then(|| &self[index])
    }
    pub(super) fn last(&self) -> Option<&T> {
        self.get(self.len.wrapping_sub(1))
    }
    pub(super) fn range(&self, range: Range<usize>) -> impl DoubleEndedIterator<Item = &T> {
        assert!(range.start <= range.end && range.end <= self.len);
        range.map(|index| &self[index])
    }
    pub(super) fn slice(&self, range: Range<usize>) -> Cow<'_, [T]> {
        assert!(range.start <= range.end && range.end <= self.len);
        if range.is_empty() {
            Cow::Borrowed(&[])
        } else if range.start / CHUNK_ITEMS == (range.end - 1) / CHUNK_ITEMS {
            let start = range.start % CHUNK_ITEMS;
            Cow::Borrowed(&self.chunks[range.start / CHUNK_ITEMS][start..start + range.len()])
        } else {
            Cow::Owned(self.range(range).cloned().collect())
        }
    }
    pub(super) fn partition_point(&self, mut predicate: impl FnMut(&T) -> bool) -> usize {
        let mut start = 0;
        let mut end = self.len;
        while start < end {
            let middle = start + (end - start) / 2;
            if predicate(&self[middle]) {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        start
    }
    pub(super) fn insert(&mut self, index: usize, mut value: T) {
        assert!(index <= self.len);
        if index == self.len {
            self.push(value);
            return;
        }
        let mut chunk = index / CHUNK_ITEMS;
        let mut offset = index % CHUNK_ITEMS;
        loop {
            if chunk == self.chunks.len() {
                self.chunks.push(Arc::new(Vec::with_capacity(CHUNK_ITEMS)));
            }
            let values = Arc::make_mut(&mut self.chunks[chunk]);
            values.insert(offset, value);
            if values.len() <= CHUNK_ITEMS {
                break;
            }
            value = values.pop().unwrap();
            chunk += 1;
            offset = 0;
        }
        self.len += 1;
    }
    pub(super) fn remove(&mut self, index: usize) {
        assert!(index < self.len);
        let chunk = index / CHUNK_ITEMS;
        Arc::make_mut(&mut self.chunks[chunk]).remove(index % CHUNK_ITEMS);
        for next in chunk + 1..self.chunks.len() {
            let value = Arc::make_mut(&mut self.chunks[next]).remove(0);
            Arc::make_mut(&mut self.chunks[next - 1]).push(value);
        }
        self.len -= 1;
        if self.len.is_multiple_of(CHUNK_ITEMS) {
            self.chunks.pop();
        }
    }
    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &T> {
        self.chunks.iter().flat_map(|chunk| chunk.iter())
    }
    pub(super) fn push(&mut self, value: T) {
        if self.len.is_multiple_of(CHUNK_ITEMS) {
            self.chunks.push(Arc::new(Vec::with_capacity(CHUNK_ITEMS)));
        }
        Arc::make_mut(self.chunks.last_mut().unwrap()).push(value);
        self.len += 1;
    }
    pub(super) fn truncate(&mut self, len: usize) {
        assert!(len <= self.len);
        self.chunks.truncate(len.div_ceil(CHUNK_ITEMS));
        if !len.is_multiple_of(CHUNK_ITEMS) {
            Arc::make_mut(self.chunks.last_mut().unwrap()).truncate(len % CHUNK_ITEMS);
        }
        self.len = len;
    }
    pub(super) fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }
}

impl<T: Clone + PartialEq> PartialEq for Chunked<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}
impl<T: Clone> Extend<T> for Chunked<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<T> Index<usize> for Chunked<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.chunks[index / CHUNK_ITEMS][index % CHUNK_ITEMS]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_edits_match_vectors_across_chunk_boundaries() {
        let mut random = fastrand::Rng::with_seed(0x7469_6d65_6c69_6e65);
        let mut actual = Chunked::default();
        let mut expected = (0..1024u32).collect::<Vec<_>>();
        actual.extend(expected.iter().copied());
        for _ in 0..128 {
            let saved = actual.clone();
            let saved_expected = expected.clone();
            for _ in 0..64 {
                match random.u8(0..10) {
                    0..=5 => {
                        let at = random.usize(0..=expected.len());
                        let value = random.u32(..);
                        actual.insert(at, value);
                        expected.insert(at, value);
                    }
                    6..=8 if !expected.is_empty() => {
                        let at = random.usize(0..expected.len());
                        actual.remove(at);
                        expected.remove(at);
                    }
                    _ => {
                        let len = expected.len().saturating_sub(random.usize(0..8));
                        actual.truncate(len);
                        expected.truncate(len);
                    }
                }
                assert_eq!(actual.to_vec(), expected);
            }
            assert_eq!(saved.to_vec(), saved_expected);
        }
        for len in [0, 1, 127, 128, 129, 255, 256, 257] {
            let mut values = Chunked::default();
            values.extend(0..len);
            for split in 0..=len {
                assert_eq!(values.partition_point(|value| *value < split), split);
                for end in split..=len {
                    assert_eq!(
                        &*values.slice(split..end),
                        &(split..end).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
