//! Append tile movement and compute operations with bound contracts.

use super::*;

impl TileGraphBuilder {
    pub(super) fn kernel_run(
        &mut self,
        provenance: WorkProvenance,
        kernel: TileKernelSpec,
        mut inputs: Vec<ShardView>,
        outputs: Vec<ShardView>,
    ) -> ExpansionResult<KernelRun> {
        for view in &mut inputs {
            self.resolve_read_view(view)?;
        }
        // Intern the contract before allocating its formats/requirements. Operand
        // views vary by tile; the kernel and storage contracts usually do not.
        let format =
            |view: &ShardView| &self.shards[view.shard.index() as usize].tensor_type.format;
        if outputs.is_empty() {
            return Err(ExpansionError::ResultArity);
        }
        for metadata in &self.kernel_metadata {
            if metadata.provenance == provenance
                && metadata.kernel == kernel
                && metadata.requirements.outputs.len() == outputs.len()
                && outputs
                    .iter()
                    .zip(&metadata.requirements.outputs)
                    .all(|(view, requirement)| *format(view) == requirement.format)
                && metadata.requirements.inputs.len() == inputs.len()
                && inputs
                    .iter()
                    .zip(&metadata.requirements.inputs)
                    .all(|(operand, requirement)| *format(operand) == requirement.format)
            {
                return Ok(KernelRun {
                    product_flops: None,
                    metadata: Arc::clone(metadata),
                    inputs,
                    outputs,
                });
            }
        }
        let requirements = KernelRequirements::new(
            &kernel,
            inputs.iter().map(|view| format(view).clone()),
            outputs.iter().map(|view| format(view).clone()),
        );
        let run = KernelRun::new(provenance, kernel, inputs, outputs, requirements);
        self.kernel_metadata.push(Arc::clone(&run.metadata));
        Ok(run)
    }

    pub(super) fn append_physical_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        self.append_ordered_phase(transfers, provenance, CopyOrder::Physical, tiles)
    }

    pub(super) fn append_ordered_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        order: CopyOrder,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        self.append_mixed_phase(std::iter::once((order, transfers)), provenance, tiles)
    }

    pub(super) fn append_mixed_phase(
        &mut self,
        mappings: impl IntoIterator<Item = (CopyOrder, BTreeMap<ShardView, Vec<ShardView>>)>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let mut transfers = Vec::new();
        for (order, mappings) in mappings {
            transfers.extend(mappings.into_iter().map(|(source, mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order,
                }
            }));
        }
        self.append_exchange_phase(transfers, provenance, tiles)
    }

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
            u32::try_from(self.phases.len()).map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.phases.push(ExchangePhase {
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
            u32::try_from(self.kernel_runs.len()).map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.kernel_runs.push(run);
        tiles
            .operations
            .push(BlockOperation::Compute { tile, run: id });
        Ok(())
    }

    pub(super) fn append_local_copy(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        copy: LocalCopy,
    ) -> ExpansionResult<()> {
        let id = LocalCopyId(
            u32::try_from(self.local_copies.len()).map_err(|_| ExpansionError::IdOverflow)?,
        );
        self.local_copies.push(copy);
        tiles
            .operations
            .push(BlockOperation::Copy { tile, copy: id });
        Ok(())
    }
}
