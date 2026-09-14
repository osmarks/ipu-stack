//! Tensor shapes, coordinate relations, layouts and resolved ownership.
//! Shared by graph semantics, planning, storage traversal and kernel binding;
//! this module describes geometry without choosing an algorithm or assigning addresses.

mod layout;
mod resolved;
pub use layout::*;
mod view;
pub use view::AxisFactorView;

/// Logical tensor dimensions. Shapes are semantic graph information; storage
/// precision and physical layout are selected during mid-level lowering.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorShape(pub Vec<u32>);

impl TensorShape {
    pub fn new(dimensions: impl IntoIterator<Item = u32>) -> Self {
        Self(dimensions.into_iter().collect())
    }

    pub fn elements(&self) -> u64 {
        self.0.iter().copied().map(u64::from).product()
    }
}

/// A right-aligned broadcast from an operand's coordinates into a result domain.
/// Borrowed dimensions keep projection cheap when binding many local fragments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Broadcast<'a> {
    input: &'a [u32],
    output: &'a [u32],
    offset: usize,
}

impl<'a> Broadcast<'a> {
    pub(crate) fn new(input: &'a [u32], output: &'a [u32]) -> Option<Self> {
        let offset = output.len().checked_sub(input.len())?;
        input
            .iter()
            .zip(&output[offset..])
            .all(|(&input, &output)| broadcast_extent(input, output) == Some(output))
            .then_some(Self {
                input,
                output,
                offset,
            })
    }

    /// An output ownership axis projects to no input axis when its coordinate
    /// is absent or broadcast. Equal dimensions retain their ownership.
    pub(crate) fn input_axis(self, output_axis: usize) -> Option<usize> {
        let input_axis = output_axis.checked_sub(self.offset)?;
        (input_axis < self.input.len() && !self.is_broadcast(input_axis)).then_some(input_axis)
    }

    pub(crate) fn output_axis(self, input_axis: usize) -> usize {
        self.offset + input_axis
    }

    pub(crate) fn is_broadcast(self, input_axis: usize) -> bool {
        self.input[input_axis] != self.output[self.output_axis(input_axis)]
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("dimensions {left} and {right} cannot be broadcast")]
pub(crate) struct BroadcastError {
    left: u32,
    right: u32,
}

fn broadcast_extent(left: u32, right: u32) -> Option<u32> {
    match (left, right) {
        (left, right) if left == right => Some(left),
        (1, right) => Some(right),
        (left, 1) => Some(left),
        _ => None,
    }
}

pub(crate) fn broadcast_shape(left: &[u32], right: &[u32]) -> Result<Vec<u32>, BroadcastError> {
    let rank = left.len().max(right.len());
    (0..rank)
        .map(|axis| {
            let dimension = |shape: &[u32]| {
                axis.checked_sub(rank - shape.len())
                    .map_or(1, |axis| shape[axis])
            };
            let (left, right) = (dimension(left), dimension(right));
            broadcast_extent(left, right).ok_or(BroadcastError { left, right })
        })
        .collect()
}
