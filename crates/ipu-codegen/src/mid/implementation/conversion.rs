//! Physical realization of selected conversion and deferred-view plans.

use super::*;

impl BlockBuilder {
    pub(super) fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<Option<Vec<BlockValueId>>> {
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
    ) -> BlockBuildResult<()> {
        if plan.output.materialization == crate::OperandMaterialization::DispatchSlices {
            let ([source], [result]) = (operation.inputs.as_slice(), operation.results.as_slice())
            else {
                return Err(BlockBuildError::InvalidConversionPlan);
            };
            if !plan.strategy.uses_intersections() {
                return Err(BlockBuildError::InvalidConversionPlan);
            }
            // Mid lowering selected consumer-sized materialization. It is not
            // contingent on where other operand conversions appear in the list.
            self.deferred_conversions.insert(*result, *source);
            for shard in self.value_shards(*result)?.to_vec() {
                self.shards[shard.index() as usize].definition = ShardDefinition::Unmaterialized;
            }
            return Ok(());
        }
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
    ) -> BlockBuildResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(BlockBuildError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
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
    ) -> BlockBuildResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(BlockBuildError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*result)?.to_vec();
        let copy_order = match plan.strategy {
            ConversionStrategy::DirectRetile => CopyOrder::Physical,
            ConversionStrategy::StageLogicalThenTransform => CopyOrder::Semantic,
            ConversionStrategy::LocalKernel => {
                return Err(BlockBuildError::InvalidConversionPlan);
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
            CopyOrder::Semantic,
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
    ) -> BlockBuildResult<()> {
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut before_exchange = Vec::new();
        let mut after_exchange = Vec::new();
        let mut after_exchange_kernels = Vec::new();
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
                        &mut before_exchange
                    } else {
                        &mut after_exchange
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
                    after_exchange_kernels.push((
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
                        &mut after_exchange,
                        CopyOrder::Semantic,
                    )?;
                }
            }
        }
        for (tile, copy) in before_exchange {
            self.append_local_copy(tiles, tile, copy)?;
        }
        self.append_ordered_phase(transfers, provenance, exchange_order, tiles)?;
        for (tile, copy) in after_exchange {
            self.append_local_copy(tiles, tile, copy)?;
        }
        for (tile, run) in after_exchange_kernels {
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
    ) -> BlockBuildResult<crate::mid::CopyPlan> {
        let shard = &self.shards[destination.index() as usize];
        let mappings = mappings
            .iter()
            .map(|(source, destination)| crate::mid::CopyMapping {
                source: self.shards[source.shard.index() as usize].storage(),
                source_extents: &source.extents,
                destination_extents: &destination.extents,
            })
            .collect::<Vec<_>>();
        Ok(crate::mid::CopyPlan::for_destination(
            &shard.tensor_type,
            &shard.extents,
            &mappings,
            copy_order,
        )?)
    }
}
