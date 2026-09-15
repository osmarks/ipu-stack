//! Distributed arithmetic and its operand windows. Tile calls are enumerated in low.

use crate::kernel::{AccumulationPrecision, GemmKernelMode};
use crate::tensor::{Precision, TensorAxis};
use serde::{Deserialize, Serialize};

/// Lifetime policy for partials reduced across a GEMM's K partitions.
#[derive(
    Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum ReductionStaging {
    /// Receive every remote partial into one packed buffer, then reduce once.
    #[default]
    Complete,
    /// Receive and accumulate one remote partial at a time. This minimizes
    /// temporary SRAM at the expense of additional exchange epochs and kernel
    /// launches.
    Streamed,
    /// Receive at most this many remote partials per exchange epoch.
    Batched(std::num::NonZeroU16),
}

impl ReductionStaging {
    pub(crate) fn remote_partials_per_stage(self, remote: u64) -> u64 {
        match self {
            Self::Complete => remote.max(1),
            Self::Streamed => 1,
            Self::Batched(limit) => u64::from(limit.get()).min(remote.max(1)),
        }
    }
}

/// A rectangular operand window in global tensor coordinates. Omitted axes
/// retain their full extent. Windows do not allocate temporary tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperandWindow(pub Vec<(u16, u32, u32)>);

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
}

impl OperandIndexing {
    pub fn local() -> Self {
        Self::Local(OperandWindow::default())
    }
}

/// Matrix axes used by a local product after the distributed operands have
/// been materialized. Product records the selected local blocking separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductAxes {
    pub left_inner: TensorAxis,
    pub right_inner: TensorAxis,
    pub output_column: TensorAxis,
    /// Valid contraction/output-column bounds when scratch shapes include padding.
    /// These describe useful arithmetic; they do not change physical execution.
    pub valid_inner: Option<u32>,
    pub valid_columns: Option<u32>,
}

/// A selected contraction over resident distributed operands. The block sizes
/// bound local calls; low clips them to each shard and binds the callable GEMM.
/// Weight load instructions depend on that binding, not this mathematical work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Product {
    pub multiply: Precision,
    pub accumulate: AccumulationPrecision,
    pub mode: GemmKernelMode,
    pub inner_block: u32,
    pub output_columns: u32,
    pub axes: ProductAxes,
    pub operands: [OperandWindow; 2],
    pub output_aliases: Vec<(usize, usize)>,
}

impl super::MidOperation {
    pub(crate) fn operand_window(&self, index: usize) -> Option<&OperandWindow> {
        if let super::MidOperationKind::Product(product) = &self.kind {
            return product.operands.get(index);
        }
        match self.operands.get(index)? {
            OperandIndexing::Local(window) => Some(window),
            OperandIndexing::Elementwise { .. } => None,
        }
    }

    pub(crate) fn input_count(&self) -> usize {
        match &self.kind {
            super::MidOperationKind::Product(product) => product.operands.len(),
            _ => self.operands.len(),
        }
    }

    pub(crate) fn output_aliases(&self) -> &[(usize, usize)] {
        match &self.kind {
            super::MidOperationKind::Product(product) => &product.output_aliases,
            _ => &self.output_aliases,
        }
    }
}
