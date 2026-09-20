//! Lower independent mid copies together: resolve regions and owners, select
//! realize their selected strategies, then emit preparation, exchange and destination work.
//! Storage supplies byte geometry; kernel::copy binds local copy launches.

use super::*;
use mapping::*;
use ownership::CopyRegions;
use realize::MaterializationBatch;

pub(super) mod mapping;
mod ownership;
pub(super) mod realize;
mod relay;

impl TileGraphBuilder {
    pub(super) fn lower_copies(
        &mut self,
        operations: &[MidOperation],
        body: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let relay = |op: &MidOperation| {
            matches!(
                op.kind,
                MidOperationKind::Copy {
                    policy: CopyPolicy::GatherThenMulticast,
                    ..
                }
            )
        };
        for operations in operations.chunk_by(|a, b| relay(a) == relay(b)) {
            let mut batch = MaterializationBatch::default();
            for operation in operations {
                self.prepare_copy_tensor(operation, &mut batch, body)?;
            }
            let first = &operations[0];
            let mut provenance = operation_provenance(first);
            if operations.len() > 1 {
                provenance.value = None;
            }
            if operations.iter().any(|next| next.source != first.source) {
                provenance.operation = None;
            }
            self.append_materialization(batch, provenance, body, relay(first))?;
        }
        Ok(())
    }
}
