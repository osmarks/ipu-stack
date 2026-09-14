//! Distributed arithmetic and its operand windows. Tile calls are enumerated in low.

use super::*;

/// A rectangular operand window in global tensor coordinates. Omitted axes
/// retain their full extent. Windows do not allocate temporary tensors.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperandWindow(pub Vec<(u16, u32, u32)>);

/// Matrix axes used by a local product after the distributed operands have
/// been materialized. Blocking is in the selected GEMM kernel specification.
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Compute {
    /// Invoke the selected kernel over the output distribution. GEMM blocking
    /// enumerates local calls later; it does not choose distribution or staging.
    Kernel {
        kernel: TileKernelSpec,
        operands: Vec<OperandWindow>,
        product: Option<ProductAxes>,
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
    /// Whole-value numerical conversion; ownership is unchanged by this call.
    pub fn cast(from: Precision, to: Precision) -> Self {
        Self::Kernel {
            kernel: TileKernelSpec::Cast { from, to },
            operands: vec![OperandWindow::default()],
            product: None,
            output_aliases: Vec::new(),
        }
    }
}
