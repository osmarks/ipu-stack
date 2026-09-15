//! Append tile movement and compute operations with bound contracts.

use super::*;

impl TileGraphBuilder {
    pub(super) fn append_exchange_phase(
        &mut self,
        transfers: Vec<LogicalExchange>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        let id = ExchangePhaseId(
            u32::try_from(self.program.exchange_phases.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.exchange_phases.push(ExchangePhase {
            id,
            provenance,
            transfers,
        });
        tracing::debug!(
            phase = id.index(),
            operation = ?provenance.operation.map(OperationId::index),
            value = ?provenance.value.map(MidValueId::index),
            reason = ?provenance.reason,
            "scheduled exchange phase"
        );
        tiles.operations.push(BlockOperation::Exchange(id));
        Ok(())
    }

    pub(super) fn append_kernel(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        run: KernelRun,
    ) -> ExpansionResult<()> {
        let id = KernelRunId(
            u32::try_from(self.program.kernel_runs.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.kernel_runs.push(run);
        tiles
            .operations
            .push(BlockOperation::Compute { tile, run: id });
        Ok(())
    }

    pub(super) fn append_local_copy(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        copy: crate::kernel::CopyRun,
    ) -> ExpansionResult<()> {
        let id = LocalCopyId(
            u32::try_from(self.program.local_copies.len())
                .map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.program.local_copies.push(copy);
        tiles
            .operations
            .push(BlockOperation::Copy { tile, copy: id });
        Ok(())
    }
}
