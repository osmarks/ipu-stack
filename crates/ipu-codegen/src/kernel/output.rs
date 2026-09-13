//! Output epilogues implemented by kernel families. Mid rewrites use these
//! contracts to move computation and its broadcast operands together.
use crate::{AmpOrder, ElementOrder, Precision, TileKernelSpec};

pub(crate) struct OutputCapability {
    /// Operand zero follows the output coordinates; the remaining operands
    /// are row parameters placed with the ordinary broadcast tiling rules.
    pub operands: usize,
    pub complete_rows: bool,
    pub input_order: ElementOrder,
    pub output_orders: &'static [ElementOrder],
    pub column_multiple: u32,
}

impl TileKernelSpec {
    pub(crate) fn output_capability(&self, precision: Precision) -> Option<OutputCapability> {
        if !matches!(precision, Precision::F8F143 { .. }) {
            return None;
        }
        let (operands, complete_rows) = match self {
            Self::Gelu => (1, false),
            Self::BiasGelu => (2, false),
            Self::LayerNorm => (3, true),
            _ => return None,
        };
        Some(OutputCapability {
            operands,
            complete_rows,
            input_order: ElementOrder::RowMajor,
            output_orders: &[ElementOrder::RowMajor, ElementOrder::Amp(AmpOrder::Left)],
            column_multiple: 4,
        })
    }
}
