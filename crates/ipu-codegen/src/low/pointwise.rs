use super::*;

impl LoweringState {
    pub(super) fn lower_schedule(
        &mut self,
        operation: &MidOperation,
        schedule: &OperatorSchedule,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [ScheduleStep::KernelMap(map)] = schedule.steps.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        if map.output != ScheduleValue::Output || map.inputs.len() != operation.inputs.len() {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
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
            let sources = map
                .inputs
                .iter()
                .map(|(value, access)| {
                    let ScheduleValue::Input(index) = value else {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    };
                    let input = operation
                        .inputs
                        .get(usize::from(*index))
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    Ok(match access {
                        ScheduleAccess::LogicalOverlap => self
                            .value_shards(*input)?
                            .iter()
                            .find_map(|source| self.broadcast_view(*source, output))
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?,
                        ScheduleAccess::TileLocal => {
                            let output_extents = &self.shards[output.index() as usize].extents;
                            let source = self
                                .value_shards(*input)?
                                .iter()
                                .copied()
                                .find(|source| {
                                    let source = &self.shards[source.index() as usize];
                                    source.tile == tile && source.extents == *output_extents
                                })
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                            self.full_view(source)
                        }
                    })
                })
                .collect::<LowLoweringResult<Vec<_>>>()?;
            let chunks = vec![self.shards[output.index() as usize].extents.clone()];
            for (wave, output_extents) in chunks.into_iter().enumerate() {
                if wave_transfers.len() <= wave {
                    wave_transfers.push(BTreeMap::new());
                    wave_runs.push(Vec::new());
                }
                let inputs = sources
                    .iter()
                    .enumerate()
                    .map(|(index, source)| {
                        let access = map.inputs[index].1;
                        let source_view = match access {
                            ScheduleAccess::LogicalOverlap => self
                                .broadcast_view_for_extents(source.shard, output, &output_extents)
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?,
                            ScheduleAccess::TileLocal => source.clone(),
                        };
                        let view = if self.shards[source_view.shard.index() as usize].tile == tile {
                            source_view
                        } else {
                            let copy = self.push_shard(LowShard {
                                id: LowShardId(0),
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
                    .collect::<LowLoweringResult<_>>()?;
                wave_runs[wave].push((
                    tile,
                    KernelRun::new(
                        WorkProvenance {
                            operation: operation.source,
                            value: operation.results.first().copied(),
                            reason: WorkReason::OperatorKernel,
                        },
                        map.kernel.clone(),
                        inputs,
                        ShardView {
                            shard: output,
                            extents: output_extents,
                        },
                        KernelRequirements::Operator(requirements.clone()),
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
                semantic_exchange,
                tiles,
            )?;
            for (tile, run) in runs {
                self.append_kernel(tiles, tile, run)?;
            }
        }
        Ok(())
    }

    fn broadcast_view(&self, source: LowShardId, output: LowShardId) -> Option<ShardView> {
        self.broadcast_view_for_extents(
            source,
            output,
            &self.shards[output.index() as usize].extents,
        )
    }

    fn broadcast_view_for_extents(
        &self,
        source: LowShardId,
        output: LowShardId,
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

    pub(super) fn schedule_input_view(
        &mut self,
        value: MidValueId,
        tile: u16,
        ranges: &[(usize, u32, u32)],
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> LowLoweringResult<ShardView> {
        let target = self.local_shard(value, tile)?;
        let target_view = self.narrow_view(target, ranges)?;
        if !self.deferred_values.contains_key(&value) {
            return Ok(target_view);
        }

        let staging = self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: self.shards[target.index() as usize].tensor_type.clone(),
            extents: target_view.extents.clone(),
            definition: ShardDefinition::ExchangeStaging,
        })?;
        self.materialize_deferred_region(
            value,
            &target_view.extents.logical(),
            staging,
            ExchangeOrder::Semantic,
            transfers,
            local_copies,
        )?;
        Ok(self.full_view(staging))
    }
}
