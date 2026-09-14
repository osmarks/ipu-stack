//! Distributed arithmetic and its operand windows. Tile calls are enumerated in low.

use super::{Precision, ReductionStaging, TensorAxis};
use crate::kernel::{AccumulationPrecision, GemmKernelMode, TileKernelSpec};

/// A rectangular operand window in global tensor coordinates. Omitted axes
/// retain their full extent. Windows do not allocate temporary tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperandWindow(pub Vec<(u16, u32, u32)>);

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Compute {
    Product(Product),
    /// Invoke the selected kernel over the output distribution.
    Kernel {
        kernel: TileKernelSpec,
        operands: Vec<OperandWindow>,
        /// (Result index, input index) pairs sharing an allocation. Inputs may
        /// include lifetime dependencies beyond the kernel's explicit operands.
        /// Shrinking casts donate storage with a displacement chosen in low;
        /// other kernels use the same byte origin.
        output_aliases: Vec<(usize, usize)>,
    },
    /// Independent partials occupy an explicit tensor axis. The selected
    /// reduction policy determines whether remote contributors arrive together
    /// or in successive bounded stages.
    Sum {
        axis: u16,
        staging: ReductionStaging,
    },
}

impl Compute {
    pub(crate) fn operand_windows(&self) -> &[OperandWindow] {
        match self {
            Self::Product(product) => &product.operands,
            Self::Kernel { operands, .. } => operands,
            Self::Sum { .. } => &[],
        }
    }

    pub(crate) fn output_aliases(&self) -> &[(usize, usize)] {
        match self {
            Self::Product(product) => &product.output_aliases,
            Self::Kernel { output_aliases, .. } => output_aliases,
            Self::Sum { .. } => &[],
        }
    }

    /// Whole-value numerical conversion; ownership is unchanged by this call.
    pub fn cast(from: Precision, to: Precision) -> Self {
        Self::Kernel {
            kernel: TileKernelSpec::Cast { from, to },
            operands: vec![OperandWindow::default()],
            output_aliases: Vec::new(),
        }
    }
}
