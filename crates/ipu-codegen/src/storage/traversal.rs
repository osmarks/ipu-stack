//! Compact byte traversals: contiguous leaves, affine repeats and short sequences.
use super::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Node {
    offset: u32,
    kind: Kind,
}
#[derive(Clone, Debug)]
enum Kind {
    Run(u32),
    Repeat {
        count: u32,
        stride: u32,
        body: Arc<Node>,
    },
    Sequence(Vec<Node>),
}
impl Node {
    fn shift(mut self, offset: u32) -> Self {
        self.offset += offset;
        self
    }
    fn repeat(self, count: u32, stride: u32) -> Self {
        if count == 1 {
            return self;
        }
        if let Kind::Run(bytes) = self.kind
            && bytes == stride
        {
            return Self {
                offset: self.offset,
                kind: Kind::Run(bytes * count),
            };
        }
        Self {
            offset: 0,
            kind: Kind::Repeat {
                count,
                stride,
                body: Arc::new(self),
            },
        }
    }
    fn sequence(mut parts: Vec<Self>) -> Self {
        if parts.len() == 1 {
            return parts.pop().unwrap();
        }
        Self {
            offset: 0,
            kind: Kind::Sequence(parts),
        }
    }
}

/// Physical traversals merge disjoint, individually sorted rectangular pieces.
/// Semantic traversals have one tree, ordered by logical tensor coordinates.
#[derive(Clone, Debug)]
pub(crate) struct ByteTraversal {
    parts: Vec<Node>,
}

#[derive(Clone, Copy)]
struct Digit {
    axis: usize,
    divisor: u32,
    count: u32,
    stride: u32,
}

fn digits(shard: TensorStorage<'_>) -> StorageResult<Vec<Digit>> {
    let widths = shard
        .extents
        .iter()
        .map(|e| e.physical_end - e.start)
        .collect::<Vec<_>>();
    let rank = widths.len();
    let mut dimensions = Vec::new();
    let mut push = |axis, divisor, count| {
        dimensions.push(Digit {
            axis,
            divisor,
            count,
            stride: 0,
        })
    };
    if shard.format.layout.order == ElementOrder::RowMajor {
        for (axis, &width) in widths.iter().enumerate() {
            push(axis, 1, width);
        }
    } else {
        if rank < 2 {
            return Err(StorageError::MatrixRank);
        }
        let transposed = matches!(
            shard.format.layout.order,
            ElementOrder::Amp(
                AmpOrder::TransposedLeft | AmpOrder::TransposedOutput | AmpOrder::TransposedRight
            ) | ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix { .. })
        );
        let (r, c) = if transposed {
            (rank - 1, rank - 2)
        } else {
            (rank - 2, rank - 1)
        };
        let (rows, cols) = (widths[r], widths[c]);
        let micro = amp_micro_dimension(shard.format.precision);
        let flatten = matches!(
            shard.format.layout.order,
            ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
        );
        if !flatten {
            for (axis, &width) in widths[..rank - 2].iter().enumerate() {
                push(axis, 1, width);
            }
        }
        match shard.format.layout.order {
            ElementOrder::Amp(
                role @ (AmpOrder::Left
                | AmpOrder::TransposedLeft
                | AmpOrder::Output
                | AmpOrder::TransposedOutput),
            ) => {
                let output = matches!(role, AmpOrder::Output | AmpOrder::TransposedOutput);
                let grain = if output { AMP_COLUMN_MICRO } else { micro };
                if !cols.is_multiple_of(grain) {
                    return Err(StorageError::AmpBlock { role });
                }
                push(c, grain, cols / grain);
                if flatten {
                    for (axis, &width) in widths[..rank - 2].iter().enumerate() {
                        push(axis, 1, width);
                    }
                }
                push(r, 1, rows);
                if output {
                    push(c, 2, 4);
                    push(c, 8, 2);
                    push(c, 1, 2);
                } else {
                    push(c, 1, grain);
                }
            }
            ElementOrder::Amp(AmpOrder::TransposedRight) => {
                if !rows.is_multiple_of(micro) || !cols.is_multiple_of(AMP_COLUMN_MICRO) {
                    return Err(StorageError::InvalidView);
                }
                push(c, AMP_COLUMN_MICRO, cols / AMP_COLUMN_MICRO);
                push(r, micro, rows / micro);
                push(c, 1, AMP_COLUMN_MICRO);
                push(r, 1, micro);
            }
            ElementOrder::BlockMajor(order) => {
                let (rb, cb) = match order {
                    BlockMajorOrder::Matrix {
                        row_block,
                        column_block,
                    }
                    | BlockMajorOrder::TransposedMatrix {
                        row_block,
                        column_block,
                    } => (u32::from(row_block), u32::from(column_block)),
                };
                if rb == 0
                    || cb == 0
                    || !rows.is_multiple_of(rb)
                    || !cols.is_multiple_of(cb)
                    || !rb.is_multiple_of(micro)
                {
                    return Err(StorageError::BlockMajorBlock { order });
                }
                push(r, rb, rows / rb);
                push(c, cb, cols / cb);
                push(r, micro, rb / micro);
                push(c, 1, cb);
                push(r, 1, micro);
            }
            ElementOrder::RowMajor => unreachable!(),
        }
    }
    let mut stride =
        u32::try_from(shard.format.precision.bytes()).map_err(|_| StorageError::Overflow)?;
    for digit in dimensions.iter_mut().rev() {
        digit.stride = stride;
        stride = stride
            .checked_mul(digit.count)
            .ok_or(StorageError::Overflow)?;
    }
    Ok(dimensions)
}

fn axis_tree(digits: &[Digit], start: u32, end: u32, suffix: &Node) -> Node {
    let Some((digit, rest)) = digits.split_first() else {
        return suffix.clone();
    };
    let div = digit.divisor;
    let first = start / div;
    let last = (end - 1) / div;
    if first == last {
        return axis_tree(rest, start % div, (end - 1) % div + 1, suffix)
            .shift(first * digit.stride);
    }
    let mut parts = Vec::new();
    if !start.is_multiple_of(div) {
        parts.push(axis_tree(rest, start % div, div, suffix).shift(first * digit.stride));
    }
    let full_start = start.div_ceil(div);
    let full_end = end / div;
    if full_start < full_end {
        parts.push(
            axis_tree(rest, 0, div, suffix)
                .repeat(full_end - full_start, digit.stride)
                .shift(full_start * digit.stride),
        );
    }
    if !end.is_multiple_of(div) {
        parts.push(axis_tree(rest, 0, end % div, suffix).shift(full_end * digit.stride));
    }
    Node::sequence(parts)
}

// A sliced logical axis is a short union of digit boxes, even when a physical
// layout permutes those digits (including AMP accumulator output pairs).
fn axis_boxes(
    digits: &[(usize, Digit)],
    start: u32,
    end: u32,
    bounds: &mut [(u32, u32)],
    visit: &mut impl FnMut(&[(u32, u32)]),
) {
    let Some((&(index, digit), rest)) = digits.split_first() else {
        visit(bounds);
        return;
    };
    let div = digit.divisor;
    let first = start / div;
    let last = (end - 1) / div;
    if first == last {
        bounds[index] = (first, 1);
        axis_boxes(rest, start % div, (end - 1) % div + 1, bounds, visit);
        return;
    }
    if !start.is_multiple_of(div) {
        bounds[index] = (first, 1);
        axis_boxes(rest, start % div, div, bounds, visit);
    }
    if !end.is_multiple_of(div) {
        bounds[index] = (end / div, 1);
        axis_boxes(rest, 0, end % div, bounds, visit);
    }
    let lo = start.div_ceil(div);
    let hi = end / div;
    if lo < hi {
        bounds[index] = (lo, hi - lo);
        for &(i, d) in rest {
            bounds[i] = (0, d.count);
        }
        visit(bounds);
    }
}

pub(crate) fn byte_traversal(
    shard: TensorStorage<'_>,
    view: &[ShardExtent],
    physical: bool,
) -> StorageResult<ByteTraversal> {
    validate_view(shard, view)?;
    if physical
        && view
            .iter()
            .zip(shard.extents)
            .all(|(a, b)| a.start == b.start && a.physical_end == b.physical_end)
    {
        return Ok(ByteTraversal::contiguous(ByteSpan {
            offset: 0,
            bytes: storage_bytes(shard)?,
        }));
    }
    if view.iter().any(|e| e.start == e.physical_end) {
        return Ok(ByteTraversal { parts: vec![] });
    }
    storage_bytes(shard)?; // All subsequent offsets and repeats lie within this u32 allocation.
    let dimensions = digits(shard)?;
    let element = shard.format.precision.bytes() as u32;
    let leaf = Node {
        offset: 0,
        kind: Kind::Run(element),
    };
    let axes = (0..view.len())
        .map(|axis| {
            let mut ds = dimensions
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, d)| d.axis == axis)
                .collect::<Vec<_>>();
            ds.sort_by_key(|(_, d)| Reverse(d.divisor));
            ds
        })
        .collect::<Vec<_>>();
    if !physical {
        let mut tree = leaf;
        for axis in (0..view.len()).rev() {
            let start = view[axis].start - shard.extents[axis].start;
            let end = view[axis].physical_end - shard.extents[axis].start;
            let ds = axes[axis].iter().map(|(_, d)| *d).collect::<Vec<_>>();
            tree = axis_tree(&ds, start, end, &tree);
        }
        return Ok(ByteTraversal { parts: vec![tree] });
    }
    let mut boxes = vec![vec![(0, 1); dimensions.len()]];
    for axis in 0..view.len() {
        let mut next = Vec::new();
        for mut bounds in boxes {
            axis_boxes(
                &axes[axis],
                view[axis].start - shard.extents[axis].start,
                view[axis].physical_end - shard.extents[axis].start,
                &mut bounds,
                &mut |bounds| next.push(bounds.to_vec()),
            );
        }
        boxes = next;
    }
    let parts = boxes
        .into_iter()
        .map(|bounds| {
            dimensions
                .iter()
                .zip(bounds)
                .rev()
                .fold(leaf.clone(), |node, (d, (start, count))| {
                    node.repeat(count, d.stride).shift(start * d.stride)
                })
        })
        .collect();
    Ok(ByteTraversal { parts })
}

struct TreeIter<'a> {
    stack: Vec<(&'a Node, u32, u32)>,
}
impl<'a> TreeIter<'a> {
    fn new(node: &'a Node) -> Self {
        Self {
            stack: vec![(node, 0, 0)],
        }
    }
}
impl Iterator for TreeIter<'_> {
    type Item = ByteSpan;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, base, index)) = self.stack.pop() {
            let base = base + node.offset;
            match &node.kind {
                Kind::Run(bytes) => {
                    return Some(ByteSpan {
                        offset: base,
                        bytes: *bytes,
                    });
                }
                Kind::Repeat {
                    count,
                    stride,
                    body,
                } => {
                    if index + 1 < *count {
                        self.stack.push((node, base - node.offset, index + 1));
                    }
                    self.stack.push((body, base + index * stride, 0));
                }
                Kind::Sequence(parts) => {
                    if let Some(part) = parts.get(index as usize) {
                        if (index as usize) + 1 < parts.len() {
                            self.stack.push((node, base - node.offset, index + 1));
                        }
                        self.stack.push((part, base, 0));
                    }
                }
            }
        }
        None
    }
}

pub(crate) struct SpanIter<'a> {
    trees: Vec<TreeIter<'a>>,
    ready: BinaryHeap<Reverse<(u32, u32, usize)>>,
    pending: Option<ByteSpan>,
}
impl ByteTraversal {
    pub(crate) fn contiguous(span: ByteSpan) -> Self {
        Self {
            parts: vec![Node {
                offset: span.offset,
                kind: Kind::Run(span.bytes),
            }],
        }
    }

    /// Merge physical address sets for coverage analysis without expanding or sorting them.
    pub(crate) fn physical_union(traversals: impl IntoIterator<Item = Self>) -> Self {
        Self {
            parts: traversals.into_iter().flat_map(|t| t.parts).collect(),
        }
    }
    pub(crate) fn spans(&self) -> SpanIter<'_> {
        let mut trees = self.parts.iter().map(TreeIter::new).collect::<Vec<_>>();
        let ready = trees
            .iter_mut()
            .enumerate()
            .filter_map(|(i, tree)| tree.next().map(|s| Reverse((s.offset, s.bytes, i))))
            .collect();
        SpanIter {
            trees,
            ready,
            pending: None,
        }
    }
    pub(crate) fn byte_len(&self) -> u64 {
        fn size(node: &Node) -> u64 {
            match &node.kind {
                Kind::Run(bytes) => u64::from(*bytes),
                Kind::Repeat { count, body, .. } => u64::from(*count) * size(body),
                Kind::Sequence(parts) => parts.iter().map(size).sum(),
            }
        }
        self.parts.iter().map(size).sum()
    }
}
impl Iterator for SpanIter<'_> {
    type Item = ByteSpan;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Reverse((offset, bytes, i))) = self.ready.pop() {
            if let Some(next) = self.trees[i].next() {
                self.ready.push(Reverse((next.offset, next.bytes, i)));
            }
            match &mut self.pending {
                Some(previous) if previous.offset + previous.bytes == offset => {
                    previous.bytes += bytes
                }
                _ => {
                    let previous = self.pending.replace(ByteSpan { offset, bytes });
                    if previous.is_some() {
                        return previous;
                    }
                }
            }
        }
        self.pending.take()
    }
}

#[derive(Clone, Copy)]
struct Summary {
    first: ByteSpan,
    last: ByteSpan,
    count: u64,
    chunks: u64,
    bad: [u64; 4],
}
fn bad(span: ByteSpan, shift: u32) -> u64 {
    u64::from((span.offset.wrapping_add(shift) | span.bytes) & 3 != 0)
}
impl Summary {
    fn span(span: ByteSpan, limit: u32) -> Self {
        Self {
            first: span,
            last: span,
            count: 1,
            chunks: u64::from(span.bytes.div_ceil(limit)),
            bad: std::array::from_fn(|i| bad(span, i as u32)),
        }
    }
    fn shift(mut self, offset: u32) -> Self {
        self.first.offset += offset;
        self.last.offset += offset;
        self.bad = std::array::from_fn(|i| self.bad[(i + offset as usize) % 4]);
        self
    }
    fn join(mut self, right: Self, limit: u32) -> Self {
        let adjacent = self.last.offset + self.last.bytes == right.first.offset;
        let merged = ByteSpan {
            offset: self.last.offset,
            bytes: if adjacent {
                self.last.bytes + right.first.bytes
            } else {
                0
            },
        };
        for i in 0..4 {
            self.bad[i] += right.bad[i];
            if adjacent {
                self.bad[i] = self.bad[i] - bad(self.last, i as u32) - bad(right.first, i as u32)
                    + bad(merged, i as u32);
            }
        }
        self.chunks += right.chunks;
        if adjacent {
            self.chunks = self.chunks
                - u64::from(self.last.bytes.div_ceil(limit))
                - u64::from(right.first.bytes.div_ceil(limit))
                + u64::from(merged.bytes.div_ceil(limit));
        }
        if adjacent && self.count == 1 {
            self.first = merged;
        }
        self.last = if adjacent && right.count == 1 {
            merged
        } else {
            right.last
        };
        self.count += right.count - u64::from(adjacent);
        self
    }
}
impl Node {
    fn summary(&self, limit: u32) -> Option<Summary> {
        fn repeated(body: Summary, count: u32, stride: u32, limit: u32) -> Summary {
            if count == 1 {
                return body;
            }
            let half = repeated(body, count / 2, stride, limit);
            let result = half.join(half.shift(count / 2 * stride), limit);
            if count.is_multiple_of(2) {
                result
            } else {
                result.join(body.shift((count - 1) * stride), limit)
            }
        }
        let result = match &self.kind {
            Kind::Run(bytes) => Some(Summary::span(
                ByteSpan {
                    offset: 0,
                    bytes: *bytes,
                },
                limit,
            )),
            Kind::Repeat {
                count,
                stride,
                body,
            } => body
                .summary(limit)
                .map(|s| repeated(s, *count, *stride, limit)),
            Kind::Sequence(parts) => parts
                .iter()
                .filter_map(|p| p.summary(limit))
                .reduce(|a, b| a.join(b, limit)),
        };
        result.map(|s| s.shift(self.offset))
    }
}
impl ByteTraversal {
    fn summary(&self, limit: u32) -> Option<Summary> {
        let mut parts = self
            .parts
            .iter()
            .filter_map(|p| p.summary(limit))
            .collect::<Vec<_>>();
        parts.sort_by_key(|s| s.first.offset);
        if parts
            .windows(2)
            .any(|p| p[0].last.offset + p[0].last.bytes > p[1].first.offset)
        {
            return self
                .spans()
                .map(|s| Summary::span(s, limit))
                .reduce(|a, b| a.join(b, limit));
        }
        parts.into_iter().reduce(|a, b| a.join(b, limit))
    }
    pub(crate) fn span_count(&self) -> u64 {
        self.summary(u32::MAX).map_or(0, |s| s.count)
    }
    pub(crate) fn contiguous_span(&self) -> Option<ByteSpan> {
        self.summary(u32::MAX)
            .filter(|s| s.count == 1)
            .map(|s| s.first)
    }
    pub(crate) fn word_aligned(&self) -> bool {
        self.summary(u32::MAX).is_none_or(|s| s.bad[0] == 0)
    }
    pub(crate) fn copy_fragments(&self, other: &Self, limit: u32) -> StorageResult<u64> {
        if self.byte_len() != other.byte_len() {
            return Err(StorageError::InvalidView);
        }
        let Some(left) = self.summary(limit) else {
            return Ok(0);
        };
        let Some(right) = other.summary(limit) else {
            return Ok(0);
        };
        if left.count == 1 {
            return Ok(right.chunks);
        }
        if right.count == 1 {
            return Ok(left.chunks);
        }
        let mut count = 0;
        crate::for_each_copy_span(self.spans(), other.spans(), |_, _, bytes| {
            count += u64::from(bytes.div_ceil(limit));
            Ok(())
        })?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_whole_allocations_retain_their_kernel_pointer_span() {
        let shard = super::super::tests::shard(crate::Layout::row_sharded(1), &[0, 16]);
        let physical = byte_traversal(shard.storage(), &shard.extents, true).unwrap();
        assert_eq!(
            physical.spans().collect::<Vec<_>>(),
            vec![ByteSpan {
                offset: 0,
                bytes: 0
            }]
        );
        assert_eq!(physical.byte_len(), 0);
        assert!(
            byte_traversal(shard.storage(), &shard.extents, false)
                .unwrap()
                .spans()
                .next()
                .is_none()
        );
    }

    #[test]
    fn compact_traversals_match_coordinate_enumeration() {
        let mut random = fastrand::Rng::with_seed(0x7472_6176_6572_7365);
        for precision in [
            Precision::F16,
            Precision::F32,
            Precision::F8F143 { scale_exponent: -4 },
        ] {
            for order in [
                ElementOrder::RowMajor,
                ElementOrder::Amp(AmpOrder::Left),
                ElementOrder::Amp(AmpOrder::Output),
                ElementOrder::Amp(AmpOrder::TransposedLeft),
                ElementOrder::Amp(AmpOrder::TransposedOutput),
                ElementOrder::Amp(AmpOrder::TransposedRight),
                ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                    row_block: 32,
                    column_block: 16,
                }),
                ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
                    row_block: 32,
                    column_block: 16,
                }),
            ] {
                let mut shard =
                    super::super::tests::shard(crate::Layout::row_sharded(1), &[3, 64, 96]);
                shard.tensor_type.format.precision = precision;
                shard.tensor_type.format.layout.order = order;
                for _ in 0..48 {
                    let view = shard
                        .extents
                        .iter()
                        .map(|e| {
                            let start = random.u32(e.start..e.physical_end);
                            let end = random.u32(start + 1..=e.physical_end);
                            ShardExtent {
                                axis: e.axis,
                                start,
                                logical_end: end,
                                physical_end: end,
                            }
                        })
                        .collect::<Vec<_>>();
                    let semantic = byte_traversal(shard.storage(), &view, false).unwrap();
                    let physical = byte_traversal(shard.storage(), &view, true).unwrap();
                    let expected = byte_spans(shard.storage(), &view, false).unwrap();
                    assert_eq!(
                        semantic.spans().collect::<Vec<_>>(),
                        expected,
                        "{precision:?} {order:?} {view:?}"
                    );
                    let mut addresses = expected
                        .iter()
                        .flat_map(|s| s.offset..s.offset + s.bytes)
                        .collect::<Vec<_>>();
                    addresses.sort_unstable();
                    assert_eq!(
                        physical
                            .spans()
                            .flat_map(|s| s.offset..s.offset + s.bytes)
                            .collect::<Vec<_>>(),
                        addresses,
                        "{precision:?} {order:?} {view:?}"
                    );
                    assert_eq!(semantic.byte_len(), addresses.len() as u64);
                    assert_eq!(physical.byte_len(), semantic.byte_len());
                    for traversal in [&semantic, &physical] {
                        let spans = traversal.spans().collect::<Vec<_>>();
                        assert_eq!(
                            traversal.word_aligned(),
                            spans.iter().all(|s| (s.offset | s.bytes) & 3 == 0)
                        );
                        for limit in [4, 28, 1024] {
                            assert_eq!(
                                traversal.summary(limit).unwrap().chunks,
                                spans
                                    .iter()
                                    .map(|s| u64::from(s.bytes.div_ceil(limit)))
                                    .sum::<u64>()
                            );
                            let mut fragments = 0;
                            crate::for_each_copy_span(
                                semantic.spans(),
                                traversal.spans(),
                                |_, _, bytes| {
                                    fragments += u64::from(bytes.div_ceil(limit));
                                    Ok(())
                                },
                            )
                            .unwrap();
                            assert_eq!(
                                semantic.copy_fragments(traversal, limit).unwrap(),
                                fragments
                            );
                        }
                    }
                }
            }
        }
    }
}
