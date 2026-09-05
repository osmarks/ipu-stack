//! Append explicit mid movement and compute operations with bound contracts.

use super::*;

impl BlockBuilder {
    pub(super) fn append_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        self.append_ordered_phase(transfers, provenance, CopyOrder::Semantic, tiles)
    }

    pub(super) fn append_physical_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        self.append_ordered_phase(transfers, provenance, CopyOrder::Physical, tiles)
    }

    pub(super) fn append_ordered_phase(
        &mut self,
        transfers: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        order: CopyOrder,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        let transfers = transfers
            .into_iter()
            .map(|(source, mut destinations)| {
                destinations.sort_unstable();
                destinations.dedup();
                LogicalExchange {
                    source,
                    destinations,
                    order,
                }
            })
            .collect::<Vec<_>>();
        self.append_exchange_phase(transfers, provenance, tiles)
    }

    pub(super) fn append_mixed_phase(
        &mut self,
        semantic: BTreeMap<ShardView, Vec<ShardView>>,
        physical: BTreeMap<ShardView, Vec<ShardView>>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        let mut transfers = Vec::with_capacity(semantic.len().saturating_add(physical.len()));
        for (order, mappings) in [
            (CopyOrder::Semantic, semantic),
            (CopyOrder::Physical, physical),
        ] {
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
        mut transfers: Vec<LogicalExchange>,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        if transfers.is_empty() {
            return Ok(());
        }
        if let Some(previous) = self.phases.last().map(|phase| phase.id)
            && self.phases[previous.index() as usize]
                .provenance
                .operation
                .is_some()
            && self.phases[previous.index() as usize].provenance.operation == provenance.operation
        {
            let touched = transfers
                .iter()
                .flat_map(|transfer| {
                    std::iter::once(transfer.source.shard)
                        .chain(transfer.destinations.iter().map(|view| view.shard))
                })
                .map(|shard| self.storage_root(shard))
                .collect::<BTreeSet<_>>();
            let previous_touched = self.phases[previous.index() as usize]
                .transfers
                .iter()
                .flat_map(|transfer| {
                    std::iter::once(transfer.source.shard)
                        .chain(transfer.destinations.iter().map(|view| view.shard))
                })
                .map(|shard| self.storage_root(shard))
                .collect::<BTreeSet<_>>();
            let disjoint_transfers = touched.is_disjoint(&previous_touched);
            let only_independent_copies_between = tiles
                .operations
                .iter()
                .rposition(|operation| *operation == BlockOperation::Exchange(previous))
                .is_some_and(|boundary| {
                    tiles.operations[boundary + 1..].iter().all(|operation| {
                        let BlockOperation::Copy { copy, .. } = operation else {
                            return false;
                        };
                        let copy = &self.local_copies[copy.0 as usize];
                        !touched.contains(&self.storage_root(copy.source))
                            && !touched.contains(&self.storage_root(copy.destination))
                    })
                });
            if disjoint_transfers && only_independent_copies_between {
                let phase = &mut self.phases[previous.index() as usize];
                phase.transfers.append(&mut transfers);
                if phase.provenance != provenance {
                    phase.provenance = WorkProvenance {
                        operation: provenance.operation,
                        value: None,
                        reason: WorkReason::OperatorInputs,
                    };
                }
                tracing::debug!(
                    phase = previous.index(),
                    operation = ?provenance.operation.map(OperationId::index),
                    "consolidated independent exchange transfers"
                );
                return Ok(());
            }
        }
        let id = ExchangePhaseId(
            u32::try_from(self.phases.len()).map_err(|_| BlockBuildError::IdOverflow)?,
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
        mut run: KernelRun,
    ) -> BlockBuildResult<()> {
        // Dispatch constraints describe a whole operator. Bind their access
        // requirements to this call's actual buffers before interning metadata.
        let requirements = &mut Arc::make_mut(&mut run.metadata).requirements;
        if run.inputs.len() > requirements.inputs.len() {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        requirements.inputs.truncate(run.inputs.len());
        for (operand, requirement) in run.inputs.iter().zip(&mut requirements.inputs) {
            let view = operand
                .views
                .first()
                .ok_or(BlockBuildError::InvalidOperatorPlan)?;
            requirement.format = self.shards[view.shard.index() as usize]
                .tensor_type
                .format
                .clone();
        }
        requirements.output.format = self.shards[run.output.shard.index() as usize]
            .tensor_type
            .format
            .clone();
        for group in &mut requirements.distinct_elements {
            group.retain(|operand| match operand {
                MemoryOperand::Output => true,
                MemoryOperand::Input(index) => usize::from(*index) < run.inputs.len(),
            });
        }
        let output_flattens_outer_rows = matches!(
            requirements.output.format.layout.order,
            ElementOrder::Amp(AmpOrder::Left | AmpOrder::Output)
        );
        if matches!(run.kernel, TileKernelSpec::Gemm { .. })
            && run.output.extents.len() > 2
            && !output_flattens_outer_rows
        {
            let matrix_axes = run.output.extents.len() - 2;
            let mut coordinates = vec![0; matrix_axes];
            let mut matrix_runs = Vec::new();
            split_gemm_matrices(&run, 0, &mut coordinates, &mut matrix_runs)?;
            if matrix_runs.len() > 1 {
                for matrix_run in matrix_runs {
                    self.append_single_kernel(tiles, tile, matrix_run)?;
                }
                return Ok(());
            }
        }
        self.append_single_kernel(tiles, tile, run)
    }

    pub(super) fn append_single_kernel(
        &mut self,
        tiles: &mut BlockRegion,
        tile: u16,
        mut run: KernelRun,
    ) -> BlockBuildResult<()> {
        if let Some(metadata) = self
            .kernel_metadata
            .iter()
            .find(|metadata| metadata.as_ref() == run.metadata.as_ref())
        {
            run.metadata = Arc::clone(metadata);
        } else {
            self.kernel_metadata.push(Arc::clone(&run.metadata));
        }
        let id = KernelRunId(
            u32::try_from(self.kernel_runs.len()).map_err(|_| BlockBuildError::IdOverflow)?,
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
    ) -> BlockBuildResult<()> {
        let id = LocalCopyId(
            u32::try_from(self.local_copies.len()).map_err(|_| BlockBuildError::IdOverflow)?,
        );
        self.local_copies.push(copy);
        tiles
            .operations
            .push(BlockOperation::Copy { tile, copy: id });
        Ok(())
    }
}
