//! Construct packed sum reductions over independent ordinary block values.
//! Contributions in a group share coordinates and element order; other layouts
//! can be supplied through explicit copies/rearrangements before this builder.

use super::*;

#[derive(Default)]
pub(super) struct SumBatch {
    stages: Vec<SumStage>,
    seeds: Vec<(u16, LocalCopy)>,
    results: Vec<(u16, LocalCopy)>,
}

#[derive(Default)]
struct SumStage {
    transfers: BTreeMap<ShardView, Vec<ShardView>>,
    runs: Vec<(u16, KernelRun)>,
}

impl TileGraphBuilder {
    pub(super) fn prepare_sum_partials(
        &mut self,
        groups: impl IntoIterator<Item = Vec<ShardView>>,
        outputs: &[BlockValueId],
        staging: crate::ReductionStaging,
        provenance: WorkProvenance,
        batch: &mut SumBatch,
    ) -> ExpansionResult<()> {
        let SumBatch {
            stages,
            seeds: seed_copies,
            results: result_copies,
        } = batch;
        let mut reduction_roots = 0usize;
        for contributors in groups {
            let Some(complete) = contributors.first() else {
                return Err(ExpansionError::InvalidOperatorPlan);
            };
            let remote_count = contributors.len() - 1;
            let remote_partials_per_stage =
                staging.remote_partials_per_stage(remote_count as u64) as usize;
            let reduction_stages = remote_count.div_ceil(remote_partials_per_stage);
            stages.resize_with(reduction_stages.max(stages.len()), SumStage::default);
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
                            &self.cache,
                            &self.shards,
                            &source,
                            &destination,
                            owner.tile,
                            result_copies,
                            CopyOrder::Physical,
                        )?;
                    } else {
                        if stages.is_empty() {
                            stages.push(SumStage::default());
                        }
                        stages[0]
                            .transfers
                            .entry(source)
                            .or_default()
                            .push(destination);
                    }
                    continue;
                }
                let seed = contributors
                    .iter()
                    .position(|view| self.shards[view.shard.index() as usize].tile == owner.tile)
                    .unwrap_or(0);
                let source_view = |partial: &ShardView| ShardView {
                    shard: partial.shard,
                    extents: intersection.clone(),
                };
                let seed_source = source_view(&contributors[seed]);
                let seed_shard = &self.shards[seed_source.shard.index() as usize];
                // A single-stage reduction only reads its accumulator. A local
                // physical slice therefore needs no packed seed buffer. Later
                // stages may overwrite it, so retain staging for those plans.
                let direct_seed = reduction_stages == 1
                    && seed_shard.tile == owner.tile
                    && view_byte_traversal(seed_shard, &seed_source, CopyOrder::Physical)?
                        .contiguous_span()
                        .is_some_and(|span| {
                            span.offset.is_multiple_of(8)
                                && u64::from(span.bytes) == u64::from(elements) * 2
                        });
                let initial = if direct_seed {
                    seed_source.shard
                } else {
                    self.push_packed_buffer(
                        owner.tile,
                        elements,
                        Precision::F16,
                        ShardDefinition::Staging,
                    )?
                };
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
                let destination = ShardView {
                    shard: output,
                    extents: intersection.clone(),
                };
                // The packed reducer can write a physical slice directly. Keep
                // aliases on the deferred-copy path: an earlier output write
                // must not overwrite a contributor needed by another group.
                let direct_output = matches!(
                    owner.definition,
                    ShardDefinition::Value(_)
                        | ShardDefinition::Staging
                        | ShardDefinition::ExchangeStaging
                ) && !contributors.iter().any(|view| view.shard == output)
                    && view_byte_traversal(&owner, &destination, CopyOrder::Physical)?
                        .contiguous_span()
                        .is_some_and(|span| {
                            span.offset.is_multiple_of(8)
                                && u64::from(span.bytes) == u64::from(elements) * 2
                        });
                let result = if direct_output && reduction_stages == 1 {
                    output
                } else {
                    self.push_packed_buffer(
                        owner.tile,
                        elements,
                        Precision::F16,
                        ShardDefinition::Staging,
                    )?
                };
                if direct_seed {
                    // The kernel reads the source view directly below.
                } else if self.shards[contributors[seed].shard.index() as usize].tile == owner.tile
                {
                    append_span_copies(
                        &self.cache,
                        &self.shards,
                        &seed_source,
                        &self.full_view(initial),
                        owner.tile,
                        seed_copies,
                        CopyOrder::Physical,
                    )?;
                } else {
                    stages[0]
                        .transfers
                        .entry(seed_source.clone())
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
                        stages[stage]
                            .transfers
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
                    stages[stage].runs.push((
                        owner.tile,
                        self.kernel_run(
                            provenance,
                            TileKernelSpec::ReductionSum {
                                partials: u16::try_from(chunk.len() + 1)
                                    .map_err(|_| ExpansionError::IdOverflow)?,
                            },
                            vec![
                                KernelOperand {
                                    views: vec![if direct_seed {
                                        seed_source.clone()
                                    } else {
                                        self.full_view(accumulator)
                                    }],
                                },
                                KernelOperand {
                                    views: vec![self.full_view(remote)],
                                },
                            ],
                            if direct_output && stage + 1 == reduction_stages {
                                destination.clone()
                            } else {
                                self.full_view(stage_result)
                            },
                        )?,
                    ));
                }
                if !direct_output {
                    let final_result = if reduction_stages.is_multiple_of(2) {
                        initial
                    } else {
                        result
                    };
                    append_span_copies(
                        &self.cache,
                        &self.shards,
                        &self.full_view(final_result),
                        &destination,
                        owner.tile,
                        result_copies,
                        CopyOrder::Physical,
                    )?;
                }
                reduction_roots += 1;
            }
            if covered != expected {
                return Err(ExpansionError::InvalidOperatorPlan);
            }
        }
        tracing::debug!(reduction_roots, "materialized packed parallel reduction");
        Ok(())
    }

    pub(super) fn append_sum_batch(
        &mut self,
        batch: SumBatch,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let SumBatch {
            stages,
            seeds: mut seed_copies,
            results: result_copies,
        } = batch;
        for (stage, SumStage { transfers, runs }) in stages.into_iter().enumerate() {
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
            crate::ReductionStaging::Batched(std::num::NonZeroU16::new(2).unwrap()),
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
            for (group, count) in [1u32, 2, 3, 4].into_iter().enumerate() {
                let start = group as u32 * 16;
                let mut block = |tile| {
                    builder
                        .push_shard(BlockValue {
                            id: BlockValueId(0),
                            tile,
                            tensor_type: TensorType {
                                shape: crate::graph::TensorShape(vec![64]),
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
                let output = block(3);
                // Leave guards on either side of the destination slice. This
                // exercises nonzero output offsets as well as whole-buffer writes.
                builder.shards[output.index() as usize].extents[0] = ShardExtent {
                    axis: 0,
                    start: 0,
                    logical_end: 64,
                    physical_end: 64,
                };
                outputs.push(output);
                groups.push(
                    sources
                        .into_iter()
                        .map(|id| builder.full_view(id))
                        .collect(),
                );
            }
            let mut region = BlockRegion::default();
            let mut batch = SumBatch::default();
            for (group, output) in groups.into_iter().zip(&outputs) {
                builder
                    .prepare_sum_partials(
                        [group],
                        &[*output],
                        staging,
                        WorkProvenance {
                            operation: None,
                            value: None,
                            reason: WorkReason::OperatorKernel,
                        },
                        &mut batch,
                    )
                    .unwrap();
            }
            builder
                .append_sum_batch(
                    batch,
                    WorkProvenance {
                        operation: None,
                        value: None,
                        reason: WorkReason::OperatorKernel,
                    },
                    &mut region,
                )
                .unwrap();
            // Interpret the expanded movement and sums, without tile lowering
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
                                    source_spans,
                                    destination_spans,
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
                            .collect::<Vec<_>>();
                        let span = view_byte_traversal(
                            &builder.shards[run.output.shard.index() as usize],
                            &run.output,
                            CopyOrder::Physical,
                        )
                        .unwrap()
                        .contiguous_span()
                        .unwrap();
                        let offset = span.offset as usize / 2;
                        memory[run.output.shard.index() as usize][offset..offset + result.len()]
                            .copy_from_slice(&result);
                    }
                    _ => panic!("unexpected sum operation"),
                }
            }
            for (group, (output, expected)) in outputs.into_iter().zip([1, 3, 6, 10]).enumerate() {
                let mut expected_values = vec![0; 64];
                expected_values[group * 16..(group + 1) * 16].fill(expected);
                assert_eq!(memory[output.index() as usize], expected_values);
                if group != 0 {
                    assert!(
                        builder
                            .kernel_runs
                            .iter()
                            .any(|run| run.output.shard == output)
                    );
                    assert!(
                        !builder
                            .local_copies
                            .iter()
                            .any(|copy| copy.destination == output)
                    );
                }
            }
        }
    }
    #[test]
    fn fragmented_sum_output_keeps_packed_result_copy() {
        let mut builder = TileGraphBuilder::new(&MidProgram {
            tile_count: 3,
            ..Default::default()
        })
        .unwrap();
        let mut block = |tile, columns| {
            builder
                .push_shard(BlockValue {
                    id: BlockValueId(0),
                    tile,
                    tensor_type: TensorType::new([2, 32], Precision::F16, Layout::row_sharded(1)),
                    extents: vec![
                        ShardExtent {
                            axis: 0,
                            start: 0,
                            logical_end: 2,
                            physical_end: 2,
                        },
                        ShardExtent {
                            axis: 1,
                            start: 0,
                            logical_end: columns,
                            physical_end: columns,
                        },
                    ],
                    definition: ShardDefinition::Staging,
                })
                .unwrap()
        };
        let left = block(0, 16);
        let right = block(1, 16);
        let output = block(2, 32);
        let mut batch = SumBatch::default();
        // The two groups cover all output columns; each is strided across rows.
        let first = vec![builder.full_view(left), builder.full_view(right)];
        let mut second = first.clone();
        for view in &mut second {
            view.extents[1].start = 16;
            view.extents[1].logical_end = 32;
            view.extents[1].physical_end = 32;
        }
        // Sources cover the complete rows; group views select column halves.
        for id in [left, right] {
            builder.shards[id.index() as usize].extents[1].logical_end = 32;
            builder.shards[id.index() as usize].extents[1].physical_end = 32;
        }
        let mut region = BlockRegion::default();
        builder
            .prepare_sum_partials(
                [first, second],
                &[output],
                ReductionStaging::Complete,
                WorkProvenance {
                    operation: None,
                    value: None,
                    reason: WorkReason::OperatorKernel,
                },
                &mut batch,
            )
            .unwrap();
        builder
            .append_sum_batch(
                batch,
                WorkProvenance {
                    operation: None,
                    value: None,
                    reason: WorkReason::OperatorKernel,
                },
                &mut region,
            )
            .unwrap();
        assert!(
            builder
                .kernel_runs
                .iter()
                .all(|run| run.output.shard != output)
        );
        assert!(
            builder
                .local_copies
                .iter()
                .any(|copy| copy.destination == output)
        );
    }
}
