//! Whole-device primitives. These describe tensor work, never tile identities.

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primitive {
    /// Populate an ordinary distributed output tensor. Each source coordinate
    /// is the corresponding output coordinate plus its selected axis offset.
    /// Storage order changes and padding are part of this materialization.
    Copy {
        offsets: Vec<u32>,
        /// Retain a compatible resident view instead of copying read-only data.
        reuse_local: bool,
    },
    View(AxisFactorView),
    /// Materialize a window of a logical view directly from its source.
    MappedCopy {
        view: AxisFactorView,
        offsets: Vec<u32>,
    },
    /// Invoke the selected kernel over the output distribution. GEMM blocking
    /// enumerates local calls later; it does not choose distribution or staging.
    Compute {
        kernel: TileKernelSpec,
        operands: Vec<OperandWindow>,
        product: Option<ProductAxes>,
        /// Input whose allocation holds the new output version. It may be an
        /// additional dependency beyond the kernel's explicit operands.
        reuse_input: Option<usize>,
    },
    /// Independent partials occupy an explicit tensor axis. The selected
    /// reduction policy determines whether remote contributors arrive together
    /// or in successive bounded stages.
    Sum {
        axis: u16,
        staging: ReductionStaging,
    },
}
