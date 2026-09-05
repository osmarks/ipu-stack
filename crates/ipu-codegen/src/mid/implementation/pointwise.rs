//! Pointwise implementation construction and broadcast views.

use super::*;

impl BlockBuilder {
    pub(super) fn build_pointwise(
        &mut self,
        operation: &MidOperation,
        kernel: TileKernelSpec,
        input_mapping: PointwiseInputMapping,
        requirements: &StorageRequirements,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        let [result] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
        };
        let outputs = self.value_shards(*result)?.to_vec();
        let mut wave_transfers = Vec::<BTreeMap<ShardView, Vec<ShardView>>>::new();
        let mut wave_runs = Vec::<Vec<(u16, KernelRun)>>::new();
        for output in outputs {
            if self.shards[output.index() as usize]
                .extents
                .iter()
                .any(|extent| extent.start == extent.physical_end)
            {
                continue;
            }
            let tile = self.shards[output.index() as usize].tile;
            let sources = operation
                .inputs
                .iter()
                .map(|input| {
                    Ok(match input_mapping {
                        PointwiseInputMapping::BroadcastToOutput => self
                            .value_shards(*input)?
                            .iter()
                            .find_map(|source| self.broadcast_view(*source, output))
                            .ok_or(BlockBuildError::InvalidOperatorPlan)?,
                        PointwiseInputMapping::TileLocal => {
                            let output_extents = &self.shards[output.index() as usize].extents;
                            let source = self
                                .value_shards(*input)?
                                .iter()
                                .copied()
                                .find(|source| {
                                    let source = &self.shards[source.index() as usize];
                                    source.tile == tile && source.extents == *output_extents
                                })
                                .ok_or(BlockBuildError::InvalidOperatorPlan)?;
                            self.full_view(source)
                        }
                    })
                })
                .collect::<BlockBuildResult<Vec<_>>>()?;
            let chunks = vec![self.shards[output.index() as usize].extents.clone()];
            for (wave, output_extents) in chunks.into_iter().enumerate() {
                if wave_transfers.len() <= wave {
                    wave_transfers.push(BTreeMap::new());
                    wave_runs.push(Vec::new());
                }
                let inputs = sources
                    .iter()
                    .map(|source| {
                        let source_view = match input_mapping {
                            PointwiseInputMapping::BroadcastToOutput => self
                                .broadcast_view_for_extents(source.shard, output, &output_extents)
                                .ok_or(BlockBuildError::InvalidOperatorPlan)?,
                            PointwiseInputMapping::TileLocal => source.clone(),
                        };
                        let view = if self.shards[source_view.shard.index() as usize].tile == tile {
                            source_view
                        } else {
                            let copy = self.push_shard(BlockValue {
                                id: BlockValueId(0),
                                tile,
                                tensor_type: self.shards[source_view.shard.index() as usize]
                                    .tensor_type
                                    .clone(),
                                extents: source_view.extents.clone(),
                                definition: ShardDefinition::ExchangeStaging,
                            })?;
                            wave_transfers[wave]
                                .entry(source_view)
                                .or_default()
                                .push(self.full_view(copy));
                            self.full_view(copy)
                        };
                        Ok(KernelOperand { views: vec![view] })
                    })
                    .collect::<BlockBuildResult<_>>()?;
                wave_runs[wave].push((
                    tile,
                    KernelRun::new(
                        WorkProvenance {
                            operation: operation.source,
                            value: operation.results.first().copied(),
                            reason: WorkReason::OperatorKernel,
                        },
                        kernel.clone(),
                        inputs,
                        ShardView {
                            shard: output,
                            extents: output_extents,
                        },
                        requirements.clone(),
                    ),
                ));
            }
        }
        for (transfers, runs) in wave_transfers.into_iter().zip(wave_runs) {
            self.append_phase(
                transfers,
                WorkProvenance {
                    operation: operation.source,
                    value: None,
                    reason: WorkReason::OperatorInputs,
                },
                tiles,
            )?;
            for (tile, run) in runs {
                self.append_kernel(tiles, tile, run)?;
            }
        }
        Ok(())
    }

    pub(super) fn broadcast_view(
        &self,
        source: BlockValueId,
        output: BlockValueId,
    ) -> Option<ShardView> {
        self.broadcast_view_for_extents(
            source,
            output,
            &self.shards[output.index() as usize].extents,
        )
    }

    pub(super) fn broadcast_view_for_extents(
        &self,
        source: BlockValueId,
        output: BlockValueId,
        output_extents: &[ShardExtent],
    ) -> Option<ShardView> {
        let source_shard = &self.shards[source.index() as usize];
        let output_shard = &self.shards[output.index() as usize];
        let source_rank = source_shard.extents.len();
        let output_rank = output_shard.extents.len();
        if source_rank > output_rank {
            return None;
        }
        let offset = output_rank - source_rank;
        let mut extents = source_shard.extents.clone();
        for (axis, extent) in extents.iter_mut().enumerate() {
            let dimension = source_shard.tensor_type.shape.0[axis];
            if dimension == 1 {
                if extent.start != 0 || extent.logical_end == 0 {
                    return None;
                }
                extent.start = 0;
                extent.logical_end = 1;
                extent.physical_end = 1;
            } else {
                let required = output_extents[offset + axis];
                if extent.start > required.start || extent.logical_end < required.logical_end {
                    return None;
                }
                extent.start = required.start;
                extent.logical_end = required.logical_end;
                extent.physical_end = required.logical_end;
            }
        }
        Some(ShardView {
            shard: source,
            extents,
        })
    }
}
