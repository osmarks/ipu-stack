//! Physical realization of selected conversion and deferred-view plans.

use super::*;

/// Work needed to populate ordinary destination blocks at one exchange boundary.
#[derive(Default)]
pub(super) struct MaterializationBatch {
    pub(super) semantic: BTreeMap<ShardView, Vec<ShardView>>,
    physical: BTreeMap<ShardView, Vec<ShardView>>,
    before: Vec<(u16, LocalCopy)>,
    pub(super) after: Vec<(u16, LocalCopy)>,
    kernels: Vec<(u16, KernelRun)>,
}

impl TileGraphBuilder {
    pub(super) fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<Option<Vec<BlockValueId>>> {
        let sources = self.value_shards(source)?.to_vec();
        for &source_shard in &sources {
            let source = &self.shards[source_shard.index() as usize];
            let compatible = source.extents.len() == 3
                && source.tensor_type.format.precision == Precision::F16
                && match source.tensor_type.format.layout.order {
                    ElementOrder::Amp(AmpOrder::Output) => {
                        let columns = source.extents[2];
                        (columns.physical_end - columns.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    ElementOrder::Amp(AmpOrder::TransposedLeft) => {
                        let rows = source.extents[1];
                        (rows.physical_end - rows.start).is_multiple_of(AMP_COLUMN_MICRO)
                    }
                    _ => false,
                };
            if !compatible {
                tracing::debug!(
                    shard = source_shard.index(),
                    rank = source.extents.len(),
                    precision = ?source.tensor_type.format.precision,
                    order = ?source.tensor_type.format.layout.order,
                    extents = ?source.extents,
                    "cannot unpack source storage into row-major order"
                );
                return Ok(None);
            }
        }

        let mut staging_shards = Vec::with_capacity(sources.len());
        for source_shard in sources {
            let source = self.shards[source_shard.index() as usize].clone();
            let mut staging_type = source.tensor_type.clone();
            staging_type.format.layout = Layout::row_major(TensorTiling::replicated(1));
            let staging = self.push_shard(BlockValue {
                id: BlockValueId(0),
                tile: source.tile,
                tensor_type: staging_type,
                extents: source.extents.clone(),
                definition: ShardDefinition::Staging,
            })?;
            self.append_kernel(
                tiles,
                source.tile,
                self.kernel_run(
                    provenance,
                    TileKernelSpec::Rearrange {
                        from: source.tensor_type.format.layout.clone(),
                        to: self.shards[staging.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .clone(),
                    },
                    vec![KernelOperand {
                        views: vec![self.full_view(source_shard)],
                    }],
                    self.full_view(staging),
                )?,
            )?;
            staging_shards.push(staging);
        }
        Ok(Some(staging_shards))
    }

    pub(super) fn build_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        match plan.strategy {
            ConversionStrategy::LocalKernel => self.build_local_conversion(operation, plan, tiles),
            ConversionStrategy::DirectRetile | ConversionStrategy::StageLogicalThenTransform => {
                self.build_intersection_conversion(operation, plan, tiles)
            }
        }
    }

    pub(super) fn build_local_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(ExpansionError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(ExpansionError::ResultArity);
        };
        for output in self.value_shards(*result)?.to_vec() {
            let tile = self.shards[output.index() as usize].tile;
            let input = self.local_shard(*input, tile)?;
            self.append_kernel(
                tiles,
                tile,
                self.kernel_run(
                    operation_provenance(operation),
                    if plan.input.format.precision != plan.output.format.precision {
                        TileKernelSpec::Cast {
                            from: plan.input.format.precision,
                            to: plan.output.format.precision,
                        }
                    } else {
                        TileKernelSpec::Rearrange {
                            from: plan.input.format.layout.clone(),
                            to: plan.output.format.layout.clone(),
                        }
                    },
                    vec![KernelOperand {
                        views: vec![self.full_view(input)],
                    }],
                    self.full_view(output),
                )?,
            )?;
        }
        Ok(())
    }

    pub(super) fn build_intersection_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(ExpansionError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(ExpansionError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*result)?.to_vec();
        let copy_order = match plan.strategy {
            ConversionStrategy::DirectRetile => CopyOrder::Physical,
            ConversionStrategy::StageLogicalThenTransform => CopyOrder::Semantic,
            ConversionStrategy::LocalKernel => {
                return Err(ExpansionError::InvalidConversionPlan);
            }
        };
        let mut mappings = Vec::new();
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            for (extents, source) in self.intersecting_shard_set(
                &inputs,
                &self.shards[output.index() as usize].extents,
                tile,
            ) {
                mappings.push((
                    ShardView {
                        shard: source,
                        extents: extents.clone(),
                    },
                    ShardView {
                        shard: output,
                        extents,
                    },
                ));
            }
        }
        self.build_mapped_views(
            mappings,
            copy_order,
            copy_order,
            operation_provenance(operation),
            tiles,
        )
    }

    pub(super) fn build_mapped_views(
        &mut self,
        mappings: Vec<(ShardView, ShardView)>,
        copy_order: CopyOrder,
        exchange_order: CopyOrder,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let physical = if copy_order == CopyOrder::Semantic {
            self.f16_micro_panel_mappings(mappings.clone())?
        } else {
            None
        };
        let (mappings, copy_order, exchange_order) = physical
            .map_or((mappings, copy_order, exchange_order), |mappings| {
                (mappings, CopyOrder::Physical, CopyOrder::Physical)
            });
        let mut batch = MaterializationBatch::default();
        self.prepare_mapped_views(
            mappings,
            copy_order,
            exchange_order,
            provenance,
            &mut batch,
            tiles,
        )?;
        self.append_materialization(batch, provenance, tiles)
    }

    pub(super) fn prepare_mapped_views(
        &mut self,
        mappings: Vec<(ShardView, ShardView)>,
        copy_order: CopyOrder,
        exchange_order: CopyOrder,
        provenance: WorkProvenance,
        batch: &mut MaterializationBatch,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let transfers = match exchange_order {
            CopyOrder::Semantic => &mut batch.semantic,
            CopyOrder::Physical => &mut batch.physical,
        };
        let mut grouped = BTreeMap::<BlockValueId, Vec<(ShardView, ShardView)>>::new();
        for mapping in mappings {
            grouped.entry(mapping.1.shard).or_default().push(mapping);
        }
        for (destination_shard, mut mappings) in grouped {
            let plan = self.copy_plan(&mappings, destination_shard, copy_order)?;
            if plan.clear_padding {
                self.append_fill_zero(tiles, destination_shard, provenance)?;
            }
            let staging = if let Some(staging) = &plan.staging {
                Some(self.push_shard(BlockValue {
                    id: BlockValueId(0),
                    tile: self.shards[destination_shard.index() as usize].tile,
                    tensor_type: staging.tensor_type.clone(),
                    extents: staging.extents.clone(),
                    definition: ShardDefinition::Staging,
                })?)
            } else {
                None
            };
            if plan.clear_padding
                && let Some(staging) = staging
            {
                self.append_fill_zero(tiles, staging, provenance)?;
            }
            for (mut source, mut destination) in mappings.drain(..) {
                if let Some(staging) = staging {
                    destination.shard = staging;
                    for extent in &mut source.extents {
                        extent.physical_end = extent.logical_end;
                    }
                    for extent in &mut destination.extents {
                        extent.physical_end = extent.logical_end;
                    }
                }
                let source_tile = self.shards[source.shard.index() as usize].tile;
                let destination_tile = self.shards[destination.shard.index() as usize].tile;
                if source_tile == destination_tile {
                    let copies = if staging.is_some() {
                        &mut batch.before
                    } else {
                        &mut batch.after
                    };
                    append_span_copies(
                        &self.shards,
                        &source,
                        &destination,
                        destination_tile,
                        copies,
                        copy_order,
                    )?;
                } else {
                    transfers.entry(source).or_default().push(destination);
                }
            }
            if let Some(staging) = staging {
                let destination = self.logical_view(destination_shard);
                let staging = self.full_view(staging);
                let tile = self.shards[destination_shard.index() as usize].tile;
                if let Some(kernel) = plan
                    .staging
                    .as_ref()
                    .and_then(|staging| staging.kernel.as_ref())
                {
                    batch.kernels.push((
                        tile,
                        self.kernel_run(
                            provenance,
                            kernel.clone(),
                            vec![KernelOperand {
                                views: vec![staging],
                            }],
                            self.full_view(destination_shard),
                        )?,
                    ));
                } else {
                    append_span_copies(
                        &self.shards,
                        &staging,
                        &destination,
                        tile,
                        &mut batch.after,
                        CopyOrder::Semantic,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn append_materialization(
        &mut self,
        batch: MaterializationBatch,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        for (tile, copy) in batch.before {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_mixed_phase(batch.semantic, batch.physical, provenance, tiles)?;
        for (tile, copy) in batch.after {
            self.append_local_copy(tiles, tile, copy)?;
        }
        for (tile, run) in batch.kernels {
            self.append_kernel(tiles, tile, run)?;
        }
        Ok(())
    }

    pub(super) fn logical_view(&self, shard: BlockValueId) -> ShardView {
        let mut view = self.full_view(shard);
        for extent in &mut view.extents {
            extent.physical_end = extent.logical_end;
        }
        view
    }

    pub(super) fn copy_plan(
        &self,
        mappings: &[(ShardView, ShardView)],
        destination: BlockValueId,
        copy_order: CopyOrder,
    ) -> ExpansionResult<crate::CopyPlan> {
        let shard = &self.shards[destination.index() as usize];
        let mappings = mappings
            .iter()
            .map(|(source, destination)| crate::CopyMapping {
                source: self.shards[source.shard.index() as usize].storage(),
                source_extents: &source.extents,
                destination_extents: &destination.extents,
            })
            .collect::<Vec<_>>();
        Ok(crate::CopyPlan::for_destination(
            &shard.tensor_type,
            &shard.extents,
            &mappings,
            copy_order,
        )?)
    }
}
