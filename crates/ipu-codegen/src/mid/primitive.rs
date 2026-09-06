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

/// Map output coordinates back to the source: first add the window offsets,
/// then apply the optional factor-axis view. Layout/storage order is separate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoordinateMapping {
    pub offsets: Vec<u32>,
    pub view: Option<AxisFactorView>,
}

impl From<AxisFactorView> for CoordinateMapping {
    fn from(view: AxisFactorView) -> Self {
        Self {
            offsets: Vec::new(),
            view: Some(view),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Primitive {
    /// Populate a distributed output using a logical coordinate mapping.
    /// Ownership, storage-order changes and padding follow the tensor layouts.
    Copy {
        mapping: CoordinateMapping,
        /// Reuse compatible resident storage when lowering can prove it safe.
        reuse_local: bool,
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
