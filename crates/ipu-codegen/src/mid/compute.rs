//! Distributed arithmetic and operand indexing shared by mid operations.

/// A rectangular operand window in global tensor coordinates. Omitted axes
/// retain their full extent. Windows do not allocate temporary tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperandWindow(pub Vec<(u16, u32, u32)>);

impl OperandWindow {
    /// Select a backed region without changing its allocation geometry.
    /// Relative windows clip at shard tails; global windows must lie inside it.
    pub(crate) fn select(
        &self,
        extents: &[crate::ShardExtent],
        relative: bool,
    ) -> Option<Vec<crate::ShardExtent>> {
        if self
            .0
            .iter()
            .any(|&(axis, _, _)| usize::from(axis) >= extents.len())
        {
            return None;
        }
        extents
            .iter()
            .map(|&e| self.select_axis(e, relative))
            .collect()
    }

    fn select_axis(&self, mut e: crate::ShardExtent, relative: bool) -> Option<crate::ShardExtent> {
        for &(_, start, end) in self.0.iter().filter(|&&(axis, _, _)| axis == e.axis) {
            if start >= end {
                return None;
            }
            let (start, end) = if relative {
                (
                    e.start.saturating_add(start).min(e.physical_end),
                    e.start.saturating_add(end).min(e.physical_end),
                )
            } else {
                if start < e.start || end > e.physical_end {
                    return None;
                }
                (start, end)
            };
            e.start = start;
            e.physical_end = end.max(start);
            e.logical_end = e.logical_end.min(e.physical_end).max(start);
        }
        Some(e)
    }

    /// Largest selected local dimensions, retaining real partition origins for
    /// global windows. This visits axis partitions, not their Cartesian product.
    pub(crate) fn local_tensor(
        &self,
        tensor: &crate::TensorType,
        relative: bool,
    ) -> Option<crate::TensorType> {
        let resolved = tensor.format.layout.resolve(&tensor.shape).ok()?;
        let shape = if let Some(axes) = resolved.axes() {
            if self
                .0
                .iter()
                .any(|&(axis, _, _)| usize::from(axis) >= axes.len())
            {
                return None;
            }
            axes.iter()
                .map(|axis| {
                    axis.partitions()
                        .iter()
                        .filter_map(|&e| self.select_axis(e, relative))
                        .map(|e| e.physical_end - e.start)
                        .max()
                })
                .collect::<Option<Vec<_>>>()?
        } else {
            let elements = u32::try_from(resolved.maximum_tile_elements()).ok()?;
            self.select(
                &[crate::ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: elements,
                    physical_end: elements,
                }],
                relative,
            )?
            .iter()
            .map(|e| e.physical_end - e.start)
            .collect()
        };
        Some(crate::TensorType {
            shape: crate::TensorShape(shape),
            format: tensor.format.clone(),
        })
    }
}

/// Logical operand selection for a distributed local kernel. The constructor
/// declares the relation; generic low binding never infers it from a kernel name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperandIndexing {
    /// Right-aligned identity/broadcast indexing relative to this result.
    /// Identity consumes complete physical panels when the selected logical
    /// fragment is unchanged. Broadcast singleton axes select one logical value.
    Elementwise { result: usize },
    /// The resident fragment corresponding to this invocation, optionally
    /// restricted by a global window. One resident fragment can serve every
    /// invocation on its owner; multiple fragments follow the result's order.
    Local(OperandWindow),
    /// Bounds relative to each resident fragment, clipped at its physical tail.
    Fragment(OperandWindow),
}

impl OperandIndexing {
    pub fn local() -> Self {
        Self::Local(OperandWindow::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_estimates_match_explicit_shard_selection() {
        use crate::{Layout, Precision, TensorAxis, TensorTiling, TensorType};
        for length in [31, 64, 65] {
            let tensor = TensorType::new(
                [length, 16],
                Precision::F16,
                Layout::row_major(TensorTiling::sharded(TensorAxis::FromStart(0), 4)),
            );
            let shards = tensor.format.layout.shard_extents(&tensor.shape).unwrap();
            for relative in [false, true] {
                for start in [0, 4, 16, 32, 64] {
                    for end in [4, 16, 32, 64, 80] {
                        let window = OperandWindow(vec![(0, start, end)]);
                        let expected = shards
                            .iter()
                            .filter_map(|(_, extents)| window.select(extents, relative))
                            .map(|extents| extents[0].physical_end - extents[0].start)
                            .max();
                        let actual = window.local_tensor(&tensor, relative).map(|t| t.shape.0[0]);
                        assert_eq!(
                            actual, expected,
                            "{length}: {start}..{end}, relative={relative}"
                        );
                    }
                }
            }
        }
        let tensor = TensorType::new([32, 16], Precision::F16, Layout::row_sharded(1));
        assert!(
            OperandWindow(vec![(0, 40, 48)])
                .local_tensor(&tensor, false)
                .is_none()
        );
        assert_eq!(
            OperandWindow(vec![(0, 24, 48)])
                .local_tensor(&tensor, true)
                .unwrap()
                .shape
                .0,
            [8, 16]
        );
    }
}
