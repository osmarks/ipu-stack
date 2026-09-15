//! Distributed arithmetic and operand indexing shared by mid operations.

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
    /// Bounds relative to each resident fragment, clipped at its physical tail.
    Fragment(OperandWindow),
}

impl OperandIndexing {
    pub fn local() -> Self {
        Self::Local(OperandWindow::default())
    }
}

impl super::MidOperation {
    pub(crate) fn operand_window(&self, index: usize) -> Option<&OperandWindow> {
        match self.operands.get(index)? {
            OperandIndexing::Local(window) | OperandIndexing::Fragment(window) => Some(window),
            OperandIndexing::Elementwise { .. } => None,
        }
    }
}
