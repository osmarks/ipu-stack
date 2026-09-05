//! Construct packed sum reductions over independent ordinary block values.
//! Contributions in a group share coordinates and element order; other layouts
//! can be supplied through explicit copies/rearrangements before this builder.

use super::*;

impl TileGraphBuilder {
    pub(super) fn append_sum_partials(
        &mut self,
        groups: impl IntoIterator<Item = Vec<ShardView>>,
        outputs: &[BlockValueId],
        staging: crate::ReductionStaging,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let mut reduction_transfers = Vec::<BTreeMap<ShardView, Vec<ShardView>>>::new();
        let mut seed_copies = Vec::<(u16, LocalCopy)>::new();
        let mut reduction_runs = Vec::<Vec<(u16, KernelRun)>>::new();
        let mut result_copies = Vec::<(u16, LocalCopy)>::new();
        let mut reduction_roots = 0usize;
        for contributors in groups {
            let Some(complete) = contributors.first() else {
                return Err(ExpansionError::InvalidOperatorPlan);
            };
            let remote_count = contributors.len() - 1;
            let remote_partials_per_stage = match staging {
                crate::ReductionStaging::Complete => remote_count.max(1),
                crate::ReductionStaging::Streamed => 1,
            };
            let reduction_stages = remote_count.div_ceil(remote_partials_per_stage);
            reduction_transfers.resize_with(
                reduction_stages.max(reduction_transfers.len()),
                BTreeMap::new,
            );
            reduction_runs.resize_with(reduction_stages.max(reduction_runs.len()), Vec::new);
            let source_order = self.shards[complete.shard.index() as usize]
                .tensor_type
                .format
                .layout
                .order;
            if contributors.iter().any(|view| {
                let format = &self.shards[view.shard.index() as usize].tensor_type.format;
                view.extents != complete.extents
                    || format.precision != Precision::F16
                    || format.layout.order != source_order
            }) {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
            let expected = complete
                .extents
                .iter()
                .try_fold(1u64, |elements, extent| {
                    elements.checked_mul(u64::from(extent.physical_end - extent.start))
                })
                .ok_or(ExpansionError::IdOverflow)?;
            let mut covered = 0u64;
            for output in outputs.iter().copied() {
                let owner = self.shards[output.index() as usize].clone();
                if owner.tensor_type.format.precision != Precision::F16
                    || owner.tensor_type.format.layout.order != source_order
                {
                    return Err(ExpansionError::InvalidOperatorPlan);
                }
                let intersection =
                    intersect_extents_with_shared_padding(&owner.extents, &complete.extents);
                let Some(intersection) = intersection else {
                    continue;
                };
                let elements = intersection
                    .iter()
                    .try_fold(1u32, |elements, extent| {
                        elements.checked_mul(extent.physical_end - extent.start)
                    })
                    .ok_or(ExpansionError::IdOverflow)?;
                if elements == 0 || !elements.is_multiple_of(8) {
                    return Err(ExpansionError::InvalidOperatorPlan);
                }
                covered = covered
                    .checked_add(u64::from(elements))
                    .ok_or(ExpansionError::IdOverflow)?;

                if remote_count == 0 {
                    let source = ShardView {
                        shard: complete.shard,
                        extents: intersection.clone(),
                    };
                    let destination = ShardView {
                        shard: output,
                        extents: intersection,
                    };
                    if self.shards[source.shard.index() as usize].tile == owner.tile {
                        append_span_copies(
                            &self.shards,
                            &source,
                            &destination,
                            owner.tile,
                            &mut result_copies,
                            CopyOrder::Physical,
                        )?;
                    } else {
                        if reduction_transfers.is_empty() {
                            reduction_transfers.push(BTreeMap::new());
                            reduction_runs.push(Vec::new());
                        }
                        reduction_transfers[0]
                            .entry(source)
                            .or_default()
                            .push(destination);
                    }
                    continue;
                }
                let initial = self.push_packed_buffer(
                    owner.tile,
                    elements,
                    Precision::F16,
                    ShardDefinition::Staging,
                )?;
                let remote_elements = elements
                    .checked_mul(
                        u32::try_from(remote_partials_per_stage)
                            .map_err(|_| ExpansionError::IdOverflow)?,
                    )
                    .ok_or(ExpansionError::IdOverflow)?;
                let remote = self.push_packed_buffer(
                    owner.tile,
                    remote_elements,
                    Precision::F16,
                    ShardDefinition::ExchangeStaging,
                )?;
                let result = self.push_packed_buffer(
                    owner.tile,
                    elements,
                    Precision::F16,
                    ShardDefinition::Staging,
                )?;
                let seed = contributors
                    .iter()
                    .position(|view| self.shards[view.shard.index() as usize].tile == owner.tile)
                    .unwrap_or(0);
                let source_view = |partial: &ShardView| ShardView {
                    shard: partial.shard,
                    extents: intersection.clone(),
                };
                let seed_source = source_view(&contributors[seed]);
                if self.shards[contributors[seed].shard.index() as usize].tile == owner.tile {
                    append_span_copies(
                        &self.shards,
                        &seed_source,
                        &self.full_view(initial),
                        owner.tile,
                        &mut seed_copies,
                        CopyOrder::Physical,
                    )?;
                } else {
                    reduction_transfers[0]
                        .entry(seed_source)
                        .or_default()
                        .push(self.full_view(initial));
                }

                let remote_contributors = contributors
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != seed)
                    .map(|(_, partial)| partial)
                    .collect::<Vec<_>>();
                for (stage, chunk) in remote_contributors
                    .chunks(remote_partials_per_stage)
                    .enumerate()
                {
                    for (slot, partial) in chunk.iter().enumerate() {
                        let start = u32::try_from(slot)
                            .map_err(|_| ExpansionError::IdOverflow)?
                            .checked_mul(elements)
                            .ok_or(ExpansionError::IdOverflow)?;
                        let end = start
                            .checked_add(elements)
                            .ok_or(ExpansionError::IdOverflow)?;
                        reduction_transfers[stage]
                            .entry(source_view(partial))
                            .or_default()
                            .push(ShardView {
                                shard: remote,
                                extents: vec![ShardExtent {
                                    axis: 0,
                                    start,
                                    logical_end: end,
                                    physical_end: end,
                                }],
                            });
                    }
                    let (accumulator, stage_result) = if stage.is_multiple_of(2) {
                        (initial, result)
                    } else {
                        (result, initial)
                    };
                    reduction_runs[stage].push((
                        owner.tile,
                        self.kernel_run(
                            provenance,
                            TileKernelSpec::ReductionSum {
                                partials: u16::try_from(chunk.len() + 1)
                                    .map_err(|_| ExpansionError::IdOverflow)?,
                            },
                            vec![
                                KernelOperand {
                                    views: vec![self.full_view(accumulator)],
                                },
                                KernelOperand {
                                    views: vec![self.full_view(remote)],
                                },
                            ],
                            self.full_view(stage_result),
                        )?,
                    ));
                }
                let final_result = if reduction_stages.is_multiple_of(2) {
                    initial
                } else {
                    result
                };
                append_span_copies(
                    &self.shards,
                    &self.full_view(final_result),
                    &ShardView {
                        shard: output,
                        extents: intersection,
                    },
                    owner.tile,
                    &mut result_copies,
                    CopyOrder::Physical,
                )?;
                reduction_roots += 1;
            }
            if covered != expected {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
        }
        for (stage, (transfers, runs)) in reduction_transfers
            .into_iter()
            .zip(reduction_runs)
            .enumerate()
        {
            self.append_physical_phase(
                transfers,
                WorkProvenance {
                    reason: WorkReason::OperatorInputs,
                    ..provenance
                },
                tiles,
            )?;
            if stage == 0 {
                for (tile, copy) in seed_copies.drain(..) {
                    self.append_local_copy(tiles, tile, copy)?;
                }
            }
            for (tile, run) in runs {
                self.append_kernel(tiles, tile, run)?;
            }
        }
        for (tile, copy) in result_copies {
            self.append_local_copy(tiles, tile, copy)?;
        }
        tracing::debug!(reduction_roots, "materialized packed parallel reduction");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_sum_groups_support_different_contributor_counts() {
        for staging in [
            crate::ReductionStaging::Complete,
            crate::ReductionStaging::Streamed,
        ] {
            let mut builder = TileGraphBuilder::new(&MidProgram {
                tile_count: 4,
                ..Default::default()
            })
            .unwrap();
            let format = crate::TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            };
            let mut groups = Vec::new();
            let mut outputs = Vec::new();
            let mut initial_values = BTreeMap::new();
            for (group, count) in [1u32, 2, 4].into_iter().enumerate() {
                let start = group as u32 * 16;
                let mut block = |tile| {
                    builder
                        .push_shard(BlockValue {
                            id: BlockValueId(0),
                            tile,
                            tensor_type: TensorType {
                                shape: crate::graph::TensorShape(vec![48]),
                                format: format.clone(),
                            },
                            extents: vec![ShardExtent {
                                axis: 0,
                                start,
                                logical_end: start + 16,
                                physical_end: start + 16,
                            }],
                            definition: ShardDefinition::Staging,
                        })
                        .unwrap()
                };
                // Place the single contributor remotely too, exercising its copy-only path.
                let sources = (0..count)
                    .map(|part| {
                        let id = block(part as u16);
                        initial_values.insert(id, part + 1);
                        id
                    })
                    .collect::<Vec<_>>();
                outputs.push(block(3));
                groups.push(
                    sources
                        .into_iter()
                        .map(|id| builder.full_view(id))
                        .collect(),
                );
            }
            let mut region = BlockRegion::default();
            builder
                .append_sum_partials(
                    groups,
                    &outputs,
                    staging,
                    WorkProvenance {
                        operation: None,
                        value: None,
                        reason: WorkReason::OperatorKernel,
                    },
                    &mut region,
                )
                .unwrap();
            // Interpret the explicit mid movement and sums, without tile lowering
            // or GEMM. Small integer values are exactly representable in FP16.
            let mut memory = builder
                .shards
                .iter()
                .map(|block| {
                    vec![
                        initial_values.get(&block.id).copied().unwrap_or(0);
                        shard_storage_bytes(block).unwrap() as usize / 2
                    ]
                })
                .collect::<Vec<_>>();
            let copy = |memory: &mut Vec<Vec<u32>>, copy: &LocalCopy| {
                assert_eq!(copy.pattern, CopyPattern::Contiguous);
                let src = copy.source_offset as usize / 2;
                let dst = copy.destination_offset as usize / 2;
                let count = copy.bytes as usize / 2;
                let values = memory[copy.source.index() as usize][src..src + count].to_vec();
                memory[copy.destination.index() as usize][dst..dst + count]
                    .copy_from_slice(&values);
            };
            for operation in &region.operations {
                match operation {
                    BlockOperation::Copy { copy: id, .. } => {
                        copy(&mut memory, &builder.local_copies[id.0 as usize])
                    }
                    BlockOperation::Exchange(id) => {
                        for transfer in &builder.phases[id.index() as usize].transfers {
                            for destination in &transfer.destinations {
                                let source_spans = view_byte_spans(
                                    &builder.shards[transfer.source.shard.index() as usize],
                                    &transfer.source,
                                )
                                .unwrap();
                                let destination_spans = view_byte_spans(
                                    &builder.shards[destination.shard.index() as usize],
                                    destination,
                                )
                                .unwrap();
                                for movement in LocalCopy::from_spans(
                                    transfer.source.shard,
                                    destination.shard,
                                    &source_spans,
                                    &destination_spans,
                                )
                                .unwrap()
                                {
                                    copy(&mut memory, &movement);
                                }
                            }
                        }
                    }
                    BlockOperation::Compute { run, .. } => {
                        let run = &builder.kernel_runs[run.0 as usize];
                        let TileKernelSpec::ReductionSum { partials } = run.kernel else {
                            panic!("unexpected compute block");
                        };
                        let initial = &memory[run.inputs[0].views[0].shard.index() as usize];
                        let remote = &memory[run.inputs[1].views[0].shard.index() as usize];
                        let result = (0..initial.len())
                            .map(|element| {
                                initial[element]
                                    + (0..usize::from(partials - 1))
                                        .map(|part| remote[part * initial.len() + element])
                                        .sum::<u32>()
                            })
                            .collect();
                        memory[run.output.shard.index() as usize] = result;
                    }
                    _ => panic!("unexpected sum operation"),
                }
            }
            for (output, expected) in outputs.into_iter().zip([1, 3, 10]) {
                assert_eq!(memory[output.index() as usize], vec![expected; 16]);
            }
        }
    }
}
