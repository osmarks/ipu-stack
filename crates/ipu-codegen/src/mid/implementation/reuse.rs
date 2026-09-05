//! Bind a retained executable fragment when its boundary ownership still matches.

use super::*;

impl BlockBuilder {
    pub(super) fn reuse_implementation(
        &mut self,
        operation: &MidOperation,
        body: &mut BlockRegion,
    ) -> BlockBuildResult<bool> {
        if operation.inputs.iter().any(|input| {
            self.has_deferred_value(*input) || self.deferred_conversions.contains_key(input)
        }) {
            return Ok(false);
        }
        let Some(&output) = operation.results.first() else {
            return Ok(false);
        };
        let boundary = operation
            .inputs
            .iter()
            .copied()
            .chain(std::iter::once(output))
            .collect::<Vec<_>>();
        let MidOperationKind::Operator {
            implementation: Some(implementation),
            ..
        } = &operation.kind
        else {
            return Ok(false);
        };
        let fragment = implementation;
        let mut ids = vec![None; fragment.shards.len()];
        for blocks in &fragment.values {
            let actual = self.value_shards(boundary[blocks.value.index() as usize])?;
            let mut by_tile = BTreeMap::<u16, Vec<BlockValueId>>::new();
            for &id in actual {
                by_tile
                    .entry(self.shards[id.index() as usize].tile)
                    .or_default()
                    .push(id);
            }
            for &id in &blocks.shards {
                let expected = &fragment.shards[id.index() as usize];
                let Some(&matching) =
                    by_tile
                        .get(&expected.tile)
                        .into_iter()
                        .flatten()
                        .find(|&&id| {
                            let block = &self.shards[id.index() as usize];
                            block.tile == expected.tile
                                && block.extents == expected.extents
                                && block.tensor_type == expected.tensor_type
                        })
                else {
                    return Ok(false);
                };
                ids[id.index() as usize] = Some(matching);
            }
        }
        let mut additional = BTreeMap::<u16, u32>::new();
        for block in &fragment.shards {
            if ids[block.id.index() as usize].is_none()
                && block.tensor_type.format.layout.memory_class == MemoryClass::Ipu21Interleaved
                && !matches!(
                    block.definition,
                    ShardDefinition::Alias(_)
                        | ShardDefinition::WritableAlias(_)
                        | ShardDefinition::ExchangeStaging
                )
            {
                let bytes = additional.entry(block.tile).or_default();
                *bytes = bytes
                    .checked_add(shard_storage_bytes(block)?)
                    .ok_or(BlockBuildError::IdOverflow)?;
            }
        }
        let used = self.interleaved_usage(0)?;
        for (tile, bytes) in additional {
            if used[usize::from(tile)].saturating_add(bytes)
                > crate::memory::IPU21_INTERLEAVED_REGION_BYTES
            {
                return Ok(false);
            }
        }
        let mut fresh = Vec::new();
        for block in &fragment.shards {
            if ids[block.id.index() as usize].is_none() {
                let id = self.push_shard(block.clone())?;
                ids[block.id.index() as usize] = Some(id);
                fresh.push(id);
            }
        }
        let ids = ids.into_iter().map(Option::unwrap).collect::<Vec<_>>();
        for id in fresh {
            let definition = &mut self.shards[id.index() as usize].definition;
            match definition {
                ShardDefinition::Alias(source)
                | ShardDefinition::WritableAlias(source)
                | ShardDefinition::LocalCopy(source) => *source = ids[source.index() as usize],
                _ => {}
            }
        }
        let map_view = |view: &ShardView| ShardView {
            shard: ids[view.shard.index() as usize],
            extents: view.extents.clone(),
        };
        let provenance = |old: WorkProvenance| WorkProvenance {
            operation: operation.source,
            value: old.value.map(|value| boundary[value.index() as usize]),
            reason: old.reason,
        };
        for work in &fragment.body.operations {
            match work {
                BlockOperation::Compute { tile, run } => {
                    let source = &fragment.kernel_runs[run.0 as usize];
                    let mut run = source.clone();
                    run.output = map_view(&source.output);
                    for operand in &mut run.inputs {
                        for view in &mut operand.views {
                            *view = map_view(view);
                        }
                    }
                    Arc::make_mut(&mut run.metadata).provenance = provenance(source.provenance);
                    self.append_single_kernel(body, *tile, run)?;
                }
                BlockOperation::Copy { tile, copy } => {
                    let mut copy = fragment.local_copies[copy.0 as usize].clone();
                    copy.source = ids[copy.source.index() as usize];
                    copy.destination = ids[copy.destination.index() as usize];
                    self.append_local_copy(body, *tile, copy)?;
                }
                BlockOperation::Exchange(phase) => {
                    let phase = &fragment.exchange_phases[phase.index() as usize];
                    let transfers = phase
                        .transfers
                        .iter()
                        .map(|transfer| LogicalExchange {
                            source: map_view(&transfer.source),
                            destinations: transfer.destinations.iter().map(&map_view).collect(),
                            order: transfer.order,
                        })
                        .collect();
                    self.append_exchange_phase(transfers, provenance(phase.provenance), body)?;
                }
                BlockOperation::Checkpoint(..) => {}
                BlockOperation::Repeat(_) => return Err(BlockBuildError::InvalidOperatorPlan),
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_fragments_match_fresh_construction() {
        let mut graph = crate::ComputeGraph::new();
        let left = graph.host_input("left", [16, 128]).unwrap();
        let right = graph.host_input("right", [128, 64]).unwrap();
        let product = graph.gemm(left, right).unwrap();
        let result = graph.gelu(product).unwrap();
        graph.set_outputs([result]).unwrap();
        let config = crate::PipelineConfig::new(4)
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
        let mut candidate =
            crate::mid::planner::lower(&graph, &config, &crate::Ipu21CostModel).unwrap();
        let retained = build_blocks(&candidate).unwrap();
        let mut builder = BlockBuilder::new(&candidate).unwrap();
        let first = candidate
            .operations
            .iter()
            .find(|operation| operation.operator_plan().is_some())
            .unwrap();
        assert!(
            builder
                .reuse_implementation(first, &mut BlockRegion::default())
                .unwrap()
        );
        for operation in &mut candidate.operations {
            if let MidOperationKind::Operator { implementation, .. } = &mut operation.kind {
                assert!(implementation.take().is_some());
            }
        }
        let rebuilt = build_blocks(&candidate).unwrap();
        assert_eq!(retained, rebuilt);
    }
}
