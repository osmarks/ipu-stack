//! Physical realization of selected conversion and deferred-view plans.

use super::*;

pub(super) struct DeferredValue {
    pub(super) transform: AxisFactorView,
    pub(super) shards: Vec<LowShardId>,
}

pub(super) fn intersect_extents(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let end = left.logical_end.min(right.logical_end);
            (start < end).then_some(ShardExtent {
                axis: left.axis,
                start,
                logical_end: end,
                physical_end: end,
            })
        })
        .collect()
}

pub(super) fn intersect_extents_with_shared_padding(
    left: &[ShardExtent],
    right: &[ShardExtent],
) -> Option<Vec<ShardExtent>> {
    if left.len() != right.len() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let start = left.start.max(right.start);
            let logical_end = left.logical_end.min(right.logical_end);
            (start < logical_end).then(|| {
                let shared_tail =
                    if logical_end == left.logical_end && logical_end == right.logical_end {
                        left.physical_end
                            .saturating_sub(left.logical_end)
                            .min(right.physical_end.saturating_sub(right.logical_end))
                    } else {
                        0
                    };
                ShardExtent {
                    axis: left.axis,
                    start,
                    logical_end,
                    physical_end: logical_end + shared_tail,
                }
            })
        })
        .collect()
}

pub(super) fn split_mapping_at_panel_boundaries(
    source_shard: &LowShard,
    mut source: ShardView,
    destination_shard: &LowShard,
    mut destination: ShardView,
) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
    let source_rank = source.extents.len();
    let destination_rank = destination.extents.len();
    let outer_elements = |extents: &[ShardExtent]| {
        extents[..extents.len().saturating_sub(2)]
            .iter()
            .try_fold(1_u32, |elements, extent| {
                elements.checked_mul(extent.logical_end - extent.start)
            })
    };
    if source_rank < 2
        || destination_rank < 2
        || source_shard.extents.len() != source_rank
        || destination_shard.extents.len() != destination_rank
        || outer_elements(&source.extents) != Some(1)
        || outer_elements(&destination.extents) != Some(1)
    {
        return Err(LowLoweringError::InvalidOperatorPlan);
    }

    let aligned_ranges = |source: ShardExtent,
                          source_shard: ShardExtent,
                          destination: ShardExtent,
                          destination_shard: ShardExtent|
     -> LowLoweringResult<Vec<(ShardExtent, ShardExtent)>> {
        let logical_width = source.logical_end - source.start;
        if logical_width != destination.logical_end - destination.start {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let width = source.physical_end - source.start;
        if width != destination.physical_end - destination.start {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let mut ranges = Vec::new();
        let mut offset = 0;
        while offset < width {
            let source_position = source
                .start
                .checked_sub(source_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let destination_position = destination
                .start
                .checked_sub(destination_shard.start)
                .and_then(|start| start.checked_add(offset))
                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
            let source_remaining = AMP_COLUMN_MICRO - source_position % AMP_COLUMN_MICRO;
            let destination_remaining = AMP_COLUMN_MICRO - destination_position % AMP_COLUMN_MICRO;
            let length = (width - offset)
                .min(source_remaining)
                .min(destination_remaining);
            let source_start = source.start + offset;
            let destination_start = destination.start + offset;
            ranges.push((
                ShardExtent {
                    axis: source.axis,
                    start: source_start,
                    logical_end: source
                        .logical_end
                        .min(source_start + length)
                        .max(source_start),
                    physical_end: source_start + length,
                },
                ShardExtent {
                    axis: destination.axis,
                    start: destination_start,
                    logical_end: destination
                        .logical_end
                        .min(destination_start + length)
                        .max(destination_start),
                    physical_end: destination_start + length,
                },
            ));
            offset += length;
        }
        Ok(ranges)
    };

    let source_row_axis = source_rank - 2;
    let source_column_axis = source_rank - 1;
    let destination_row_axis = destination_rank - 2;
    let destination_column_axis = destination_rank - 1;

    // The global row tail can finish part-way through a micro-panel while
    // both allocations contain padding through the same panel boundary.
    // Carry that padding with the useful values so the direct physical
    // exchange remains word-aligned. A split head's column tail is not
    // extended because the following source columns may belong to another
    // head rather than padding.
    let source_rows = source.extents[source_row_axis];
    let destination_rows = destination.extents[destination_row_axis];
    if source_rows.logical_end == source_shard.tensor_type.shape.0[source_row_axis]
        && destination_rows.logical_end
            == destination_shard.tensor_type.shape.0[destination_row_axis]
    {
        let source_panel_tail = (AMP_COLUMN_MICRO
            - (source_rows.logical_end - source_shard.extents[source_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let destination_panel_tail = (AMP_COLUMN_MICRO
            - (destination_rows.logical_end
                - destination_shard.extents[destination_row_axis].start)
                % AMP_COLUMN_MICRO)
            % AMP_COLUMN_MICRO;
        let padding = source_panel_tail
            .min(destination_panel_tail)
            .min(source_shard.extents[source_row_axis].physical_end - source_rows.logical_end)
            .min(
                destination_shard.extents[destination_row_axis].physical_end
                    - destination_rows.logical_end,
            );
        source.extents[source_row_axis].physical_end += padding;
        destination.extents[destination_row_axis].physical_end += padding;
    }

    let rows = aligned_ranges(
        source.extents[source_row_axis],
        source_shard.extents[source_row_axis],
        destination.extents[destination_row_axis],
        destination_shard.extents[destination_row_axis],
    )?;
    let columns = aligned_ranges(
        source.extents[source_column_axis],
        source_shard.extents[source_column_axis],
        destination.extents[destination_column_axis],
        destination_shard.extents[destination_column_axis],
    )?;
    let mut pieces = Vec::with_capacity(rows.len().saturating_mul(columns.len()));
    for (source_row, destination_row) in rows {
        for &(source_column, destination_column) in &columns {
            let mut source_extents = source.extents.clone();
            let mut destination_extents = destination.extents.clone();
            source_extents[source_row_axis] = source_row;
            source_extents[source_column_axis] = source_column;
            destination_extents[destination_row_axis] = destination_row;
            destination_extents[destination_column_axis] = destination_column;
            pieces.push((
                ShardView {
                    shard: source.shard,
                    extents: source_extents,
                },
                ShardView {
                    shard: destination.shard,
                    extents: destination_extents,
                },
            ));
        }
    }
    Ok(pieces)
}

impl LoweringState {
    pub(super) fn deferred_root(&self, mut value: MidValueId) -> Option<MidValueId> {
        let mut remaining = self.deferred_conversions.len().saturating_add(1);
        while !self.deferred_values.contains_key(&value) {
            value = *self.deferred_conversions.get(&value)?;
            remaining = remaining.checked_sub(1)?;
        }
        Some(value)
    }

    pub(super) fn has_deferred_value(&self, value: MidValueId) -> bool {
        self.deferred_root(value).is_some()
    }

    pub(super) fn deferred_supports_physical_exchange(
        &self,
        value: MidValueId,
        destination: LowShardId,
    ) -> bool {
        let Some(root) = self.deferred_root(value) else {
            return false;
        };
        let Some(source) = self
            .deferred_values
            .get(&root)
            .and_then(|deferred| deferred.shards.first())
        else {
            return false;
        };
        self.value_shards(value)
            .ok()
            .and_then(|shards| shards.first())
            .and_then(|shard| {
                self.shards[shard.index() as usize]
                    .tensor_type
                    .shape
                    .0
                    .last()
            })
            .is_some_and(|width| width.is_multiple_of(2))
            && self.shards[source.index() as usize]
                .tensor_type
                .format
                .supports_f16_micro_panel_exchange(
                    &self.shards[destination.index() as usize].tensor_type.format,
                )
    }

    pub(super) fn unpack_amp_to_row_major(
        &mut self,
        source: MidValueId,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<Option<Vec<LowShardId>>> {
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
            let staging = self.push_shard(LowShard {
                id: LowShardId(0),
                tile: source.tile,
                tensor_type: staging_type,
                extents: source.extents.clone(),
                definition: ShardDefinition::Staging,
            })?;
            self.append_kernel(
                tiles,
                source.tile,
                KernelRun::new(
                    provenance,
                    TileKernel::Planned(TileKernelSpec::Rearrange {
                        from: source.tensor_type.format.layout.clone(),
                        to: self.shards[staging.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .clone(),
                    }),
                    vec![KernelOperand {
                        views: vec![self.full_view(source_shard)],
                    }],
                    self.full_view(staging),
                    KernelRequirements::Conversion {
                        input: OperandRequirement::new(source.tensor_type.format, 4),
                        output: OperandRequirement::new(
                            self.shards[staging.index() as usize]
                                .tensor_type
                                .format
                                .clone(),
                            4,
                        ),
                        distinct_elements: Vec::new(),
                    },
                ),
            )?;
            staging_shards.push(staging);
        }
        Ok(Some(staging_shards))
    }

    pub(super) fn lower_deferred_output(
        &mut self,
        operation: &MidOperation,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<bool> {
        let Some(offered) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
        else {
            return Ok(false);
        };
        let [result] = operation.results.as_slice() else {
            return Ok(false);
        };
        let Some(source) = operation.inputs.get(offered.source_input) else {
            return Ok(false);
        };
        let source_type = &self.shards[self.value_shards(*source)?[0].index() as usize].tensor_type;
        let result_type = &self.shards[self.value_shards(*result)?[0].index() as usize].tensor_type;
        if offered.transform.output_shape(&source_type.shape).as_ref() != Some(&result_type.shape) {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let source_shards = self.value_shards(*source)?.to_vec();
        let source_format = &self.shards[source_shards[0].index() as usize]
            .tensor_type
            .format;
        let result_format = &self.shards[self.value_shards(*result)?[0].index() as usize]
            .tensor_type
            .format;
        let direct_panel_exchange = source_format.supports_f16_micro_panel_exchange(result_format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?source_format.layout.order,
            result_order = ?result_format.layout.order,
            direct_panel_exchange,
            "selected deferred-output storage"
        );
        let staging_shards = if direct_panel_exchange
            || !matches!(
                source_format.layout.order,
                ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
            ) {
            source_shards
        } else {
            let Some(staging) = self.unpack_amp_to_row_major(
                *source,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            else {
                return Ok(false);
            };
            staging
        };
        self.deferred_values.insert(
            *result,
            DeferredValue {
                transform: offered.transform,
                shards: staging_shards,
            },
        );
        for shard in self.value_shards(*result)?.to_vec() {
            self.shards[shard.index() as usize].definition = ShardDefinition::Unmaterialized;
        }
        Ok(true)
    }

    pub(super) fn lower_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if plan.output.materialization == crate::OperandMaterialization::DispatchSlices {
            let ([source], [result]) = (operation.inputs.as_slice(), operation.results.as_slice())
            else {
                return Err(LowLoweringError::InvalidConversionPlan);
            };
            if !plan.strategy.uses_intersections() {
                return Err(LowLoweringError::InvalidConversionPlan);
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
            ConversionStrategy::LocalKernel => self.lower_local_conversion(operation, plan, tiles),
            ConversionStrategy::DirectRetile | ConversionStrategy::StageLogicalThenTransform => {
                self.lower_intersection_conversion(operation, plan, tiles)
            }
        }
    }

    pub(super) fn lower_local_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        for output in self.value_shards(*result)?.to_vec() {
            let tile = self.shards[output.index() as usize].tile;
            let input = self.local_shard(*input, tile)?;
            self.append_kernel(
                tiles,
                tile,
                KernelRun::new(
                    operation_provenance(operation),
                    TileKernel::Planned(
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
                    ),
                    vec![KernelOperand {
                        views: vec![self.full_view(input)],
                    }],
                    self.full_view(output),
                    KernelRequirements::Conversion {
                        input: plan.input.clone(),
                        output: plan.output.clone(),
                        distinct_elements: Vec::new(),
                    },
                ),
            )?;
        }
        Ok(())
    }

    pub(super) fn lower_intersection_conversion(
        &mut self,
        operation: &MidOperation,
        plan: &crate::ConversionPlan,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidConversionPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let inputs = self.value_shards(*input)?.to_vec();
        let outputs = self.value_shards(*result)?.to_vec();
        let logical_order = match plan.strategy {
            ConversionStrategy::DirectRetile => false,
            ConversionStrategy::StageLogicalThenTransform => true,
            ConversionStrategy::LocalKernel => {
                return Err(LowLoweringError::InvalidConversionPlan);
            }
        };
        let mut mappings = Vec::new();
        for output in outputs {
            let tile = self.shards[output.index() as usize].tile;
            let mut unique_intersections = BTreeMap::<Vec<ShardExtent>, LowShardId>::new();
            for source in &inputs {
                let Some(extents) = intersect_extents(
                    &self.shards[source.index() as usize].extents,
                    &self.shards[output.index() as usize].extents,
                ) else {
                    continue;
                };
                let selected = unique_intersections.entry(extents).or_insert(*source);
                if self.shards[source.index() as usize].tile == tile {
                    *selected = *source;
                }
            }
            for (extents, source) in unique_intersections {
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
        self.lower_mapped_views(
            mappings,
            logical_order,
            ExchangeOrder::Semantic,
            operation_provenance(operation),
            tiles,
        )
    }

    pub(super) fn lower_mapped_views(
        &mut self,
        mappings: Vec<(ShardView, ShardView)>,
        logical_order: bool,
        exchange_order: ExchangeOrder,
        provenance: WorkProvenance,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
        let mut before_exchange = Vec::new();
        let mut after_exchange = Vec::new();
        let mut after_exchange_kernels = Vec::new();
        let mut grouped = BTreeMap::<LowShardId, Vec<(ShardView, ShardView)>>::new();
        for mapping in mappings {
            grouped.entry(mapping.1.shard).or_default().push(mapping);
        }
        for (destination_shard, mut mappings) in grouped {
            let destination_format = &self.shards[destination_shard.index() as usize]
                .tensor_type
                .format;
            let destination_unaligned = mappings.iter().try_fold(
                false,
                |unaligned, (_, destination)| -> LowLoweringResult<bool> {
                    let spans = logical_view_byte_spans(
                        &self.shards[destination.shard.index() as usize],
                        destination,
                    )?;
                    Ok(unaligned
                        || spans
                            .iter()
                            .any(|span| span.offset & 0b11 != 0 || span.bytes & 0b11 != 0))
                },
            )?;
            let requires_materialization = logical_order
                && (mappings.iter().any(|(source, _)| {
                    self.shards[source.shard.index() as usize]
                        .tensor_type
                        .format
                        .layout
                        .order
                        != destination_format.layout.order
                }) || destination_unaligned);
            let direct_logical = requires_materialization
                && self.mappings_benefit_from_word_exchange(&mappings, destination_shard)?;
            let stage_destination = requires_materialization && !direct_logical;
            if direct_logical && self.shard_has_padding(destination_shard) {
                self.append_fill_zero(tiles, destination_shard, provenance)?;
            }
            let staging = if stage_destination {
                Some(self.push_conversion_staging(destination_shard)?)
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
                    if logical_order {
                        append_logical_span_copies(
                            &self.shards,
                            &source,
                            &destination,
                            destination_tile,
                            copies,
                        )?;
                    } else {
                        append_span_copies(
                            &self.shards,
                            &source,
                            &destination,
                            destination_tile,
                            copies,
                        )?;
                    }
                } else {
                    transfers.entry(source).or_default().push(destination);
                }
            }
            if let Some(staging) = staging {
                let destination = self.logical_view(destination_shard);
                let staging = self.full_view(staging);
                let source_format = self.shards[staging.shard.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let destination_format = self.shards[destination_shard.index() as usize]
                    .tensor_type
                    .format
                    .clone();
                let tile = self.shards[destination_shard.index() as usize].tile;
                if source_format.precision == crate::Precision::F16
                    && source_format.layout.order == ElementOrder::RowMajor
                    && matches!(
                        destination_format.layout.order,
                        ElementOrder::Amp(AmpOrder::Left | AmpOrder::TransposedRight)
                            | ElementOrder::BlockMajor(BlockMajorOrder::Matrix { .. })
                    )
                {
                    after_exchange_kernels.push((
                        tile,
                        KernelRun::new(
                            provenance,
                            TileKernel::Planned(TileKernelSpec::Rearrange {
                                from: source_format.layout.clone(),
                                to: destination_format.layout.clone(),
                            }),
                            vec![KernelOperand {
                                views: vec![staging],
                            }],
                            self.full_view(destination_shard),
                            KernelRequirements::Conversion {
                                input: OperandRequirement::new(source_format, 2),
                                output: OperandRequirement::new(destination_format, 2),
                                distinct_elements: vec![vec![
                                    MemoryOperand::Input(0),
                                    MemoryOperand::Output,
                                ]],
                            },
                        ),
                    ));
                } else {
                    append_logical_span_copies(
                        &self.shards,
                        &staging,
                        &destination,
                        tile,
                        &mut after_exchange,
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

    pub(super) fn push_conversion_staging(
        &mut self,
        destination: LowShardId,
    ) -> LowLoweringResult<LowShardId> {
        let destination = &self.shards[destination.index() as usize];
        let mut extents = destination.extents.clone();
        let tile = destination.tile;
        let shape = destination.tensor_type.shape.clone();
        let precision = destination.tensor_type.format.precision;
        for extent in &mut extents {
            extent.physical_end = extent.logical_end;
        }
        self.push_shard(LowShard {
            id: LowShardId(0),
            tile,
            tensor_type: TensorType {
                shape,
                format: crate::TensorFormat {
                    precision,
                    layout: Layout {
                        order: ElementOrder::RowMajor,
                        tiling: TensorTiling::replicated(1),
                        memory_class: MemoryClass::Ipu21Standard,
                    },
                },
            },
            extents,
            definition: ShardDefinition::Staging,
        })
    }

    pub(super) fn logical_view(&self, shard: LowShardId) -> ShardView {
        let mut view = self.full_view(shard);
        for extent in &mut view.extents {
            extent.physical_end = extent.logical_end;
        }
        view
    }

    pub(super) fn lower_view(
        &mut self,
        operation: &MidOperation,
        operator: &crate::MidOperator,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let crate::MidOperator::View(view) = operator else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [input] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [result] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let input_type = self
            .value_shards(*input)?
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let output_shards = self.value_shards(*result)?.to_vec();
        let output_type = output_shards
            .first()
            .map(|shard| self.shards[shard.index() as usize].tensor_type.clone())
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let original_sources = self.value_shards(*input)?.to_vec();
        let direct_panel_exchange = input_type
            .format
            .supports_f16_micro_panel_exchange(&output_type.format);
        tracing::debug!(
            source = ?operation.source.map(OperationId::index),
            source_order = ?input_type.format.layout.order,
            destination_order = ?output_type.format.layout.order,
            direct_panel_exchange,
            "selected view exchange strategy"
        );
        if direct_panel_exchange {
            let mappings = self.view_mappings(&original_sources, &output_shards, *view)?;
            if let Some(mappings) = self.f16_micro_panel_mappings(mappings)? {
                tracing::info!(
                    source = ?operation.source.map(OperationId::index),
                    mappings = mappings.len(),
                    "lowering view as physical micro-panel exchange"
                );
                return self.lower_mapped_views(
                    mappings,
                    false,
                    ExchangeOrder::Physical,
                    WorkProvenance {
                        operation: operation.source,
                        value: Some(*result),
                        reason: WorkReason::OperatorInputs,
                    },
                    tiles,
                );
            }
        }

        let source_shards = if matches!(
            input_type.format.layout.order,
            ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedLeft)
        ) {
            self.unpack_amp_to_row_major(
                *input,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*result),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?
            .ok_or(LowLoweringError::InvalidOperatorPlan)?
        } else {
            self.value_shards(*input)?.to_vec()
        };

        let mappings = self.view_mappings(&source_shards, &output_shards, *view)?;
        self.lower_mapped_views(
            mappings,
            true,
            ExchangeOrder::Semantic,
            WorkProvenance {
                operation: operation.source,
                value: Some(*result),
                reason: WorkReason::OperatorInputs,
            },
            tiles,
        )
    }

    pub(super) fn view_mappings(
        &self,
        source_shards: &[LowShardId],
        output_shards: &[LowShardId],
        view: AxisFactorView,
    ) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
        let mut mappings = Vec::new();
        for &output in output_shards {
            let output_extents = self.shards[output.index() as usize].extents.clone();
            let tile = self.shards[output.index() as usize].tile;
            let source_shape = &self.shards[source_shards[0].index() as usize]
                .tensor_type
                .shape;
            let output_shape = &self.shards[output.index() as usize].tensor_type.shape;
            if view.output_shape(source_shape).as_ref() != Some(output_shape) {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            let split = view.split_axis;
            let merge = view.merge_axis;
            let part_width = output_shape.0[split];
            for stream in output_extents[merge].start..output_extents[merge].logical_end {
                let mut stream_extents = output_extents.clone();
                stream_extents[merge].start = stream;
                stream_extents[merge].logical_end = stream + 1;
                stream_extents[merge].physical_end = stream + 1;
                let (target, column_base) = view
                    .source_extents(source_shape, output_shape, &stream_extents)
                    .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                for (mut source_extents, source) in
                    self.intersecting_shard_set(source_shards, &target, tile)
                {
                    let mut destination_extents = source_extents.clone();
                    destination_extents[merge] = stream_extents[merge];
                    destination_extents[split].start -= column_base;
                    destination_extents[split].logical_end -= column_base;
                    destination_extents[split].physical_end -= column_base;
                    let source_shard = &self.shards[source.index() as usize];
                    let complete_part = source_extents[split].start == column_base
                        && source_extents[split].logical_end == column_base + part_width
                        && source_shard.extents[split].start == column_base
                        && source_shard.extents[split].logical_end == column_base + part_width;
                    if complete_part {
                        let source_padding = source_shard.extents[split]
                            .physical_end
                            .saturating_sub(source_extents[split].logical_end);
                        let destination_padding = output_extents[split]
                            .physical_end
                            .saturating_sub(output_extents[split].logical_end);
                        let padding = source_padding.min(destination_padding);
                        source_extents[split].physical_end += padding;
                        destination_extents[split].physical_end += padding;
                    }
                    let source_view = ShardView {
                        shard: source,
                        extents: source_extents,
                    };
                    let destination_view = ShardView {
                        shard: output,
                        extents: destination_extents,
                    };
                    mappings.push((source_view, destination_view));
                }
            }
        }
        Ok(mappings)
    }

    /// Splits corresponding views at each allocation's F16 micro-panel
    /// boundaries. Within every resulting rectangle the source and
    /// destination have identical physical traversal, even when their outer
    /// panel sequence and tile ownership differ.
    pub(super) fn f16_micro_panel_mappings(
        &self,
        mappings: Vec<(ShardView, ShardView)>,
    ) -> LowLoweringResult<Option<Vec<(ShardView, ShardView)>>> {
        let mut split = Vec::new();
        for (source, destination) in mappings {
            let source_shard = &self.shards[source.shard.index() as usize];
            let destination_shard = &self.shards[destination.shard.index() as usize];
            let pieces = split_mapping_at_panel_boundaries(
                source_shard,
                source,
                destination_shard,
                destination,
            )?;
            for (source, destination) in pieces {
                let source_spans = view_byte_spans(source_shard, &source)?;
                let destination_spans = view_byte_spans(destination_shard, &destination)?;
                let valid_spans = source_spans
                    .iter()
                    .chain(&destination_spans)
                    .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0);
                let source_bytes = source_spans.iter().map(|span| span.bytes).sum::<u32>();
                let destination_bytes =
                    destination_spans.iter().map(|span| span.bytes).sum::<u32>();
                if !valid_spans || source_bytes != destination_bytes {
                    return Ok(None);
                }
                split.push((source, destination));
            }
        }
        Ok(Some(split))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn deferred_panel_mappings(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
    ) -> LowLoweringResult<Vec<(ShardView, ShardView)>> {
        let deferred_root = self
            .deferred_root(value)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let deferred = self
            .deferred_values
            .get(&deferred_root)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let deferred_shards = deferred.shards.clone();
        let logical_type = &self.shards[self.value_shards(value)?[0].index() as usize].tensor_type;
        let source_type = &self.shards[deferred.shards[0].index() as usize].tensor_type;
        let logical_target = [
            (stream, stream + 1),
            (row_start, row_start + rows),
            (column_start, column_start + columns),
        ];
        let mapping = deferred
            .transform
            .map_slice(&source_type.shape, &logical_type.shape, &logical_target)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let target = mapping
            .source_ranges
            .iter()
            .copied()
            .enumerate()
            .map(|(axis, (start, end))| {
                Ok(ShardExtent {
                    axis: u16::try_from(axis).map_err(|_| LowLoweringError::IdOverflow)?,
                    start,
                    logical_end: end,
                    physical_end: end,
                })
            })
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mut covered = 0u64;
        let mut mappings = Vec::new();
        for (source_extents, source) in
            self.intersecting_shard_set(&deferred_shards, &target, destination_tile)
        {
            let destination_extents = mapping
                .destination_source_axes
                .iter()
                .enumerate()
                .map(|(destination_axis, &source_axis)| {
                    let source = source_extents
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                    let base = target
                        .get(source_axis)
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?
                        .start;
                    Ok(ShardExtent {
                        axis: u16::try_from(destination_axis)
                            .map_err(|_| LowLoweringError::IdOverflow)?,
                        start: source.start - base,
                        logical_end: source.logical_end - base,
                        physical_end: source.logical_end - base,
                    })
                })
                .collect::<LowLoweringResult<Vec<_>>>()?;
            covered = covered.saturating_add(
                u64::from(source_extents[1].logical_end - source_extents[1].start)
                    * u64::from(source_extents[2].logical_end - source_extents[2].start),
            );
            let source_view = ShardView {
                shard: source,
                extents: source_extents,
            };
            let destination_view = ShardView {
                shard: destination,
                extents: destination_extents,
            };
            mappings.push((source_view, destination_view));
        }
        if covered != u64::from(rows) * u64::from(columns) {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        Ok(mappings)
    }

    pub(super) fn mapping_word_exchange_fragments(
        &self,
        mappings: &[(ShardView, ShardView)],
    ) -> LowLoweringResult<Option<u64>> {
        let maximum_bytes = ipu_exchange::MAX_TRANSFER_WORDS
            .checked_mul(4)
            .ok_or(LowLoweringError::IdOverflow)?;
        let mut fragments = 0u64;
        for (source, destination) in mappings {
            let source_spans =
                logical_view_byte_spans(&self.shards[source.shard.index() as usize], source)?;
            let destination_spans = logical_view_byte_spans(
                &self.shards[destination.shard.index() as usize],
                destination,
            )?;
            let aligned = source_spans
                .iter()
                .chain(&destination_spans)
                .all(|span| span.offset & 0b11 == 0 && span.bytes & 0b11 == 0);
            let source_bytes = source_spans.iter().map(|span| span.bytes).sum::<u32>();
            let destination_bytes = destination_spans.iter().map(|span| span.bytes).sum::<u32>();
            if !aligned || source_bytes != destination_bytes {
                tracing::trace!(
                    source = ?source,
                    destination = ?destination,
                    source_order = ?self.shards[source.shard.index() as usize]
                        .tensor_type.format.layout.order,
                    destination_order = ?self.shards[destination.shard.index() as usize]
                        .tensor_type.format.layout.order,
                    source_spans = ?source_spans,
                    destination_spans = ?destination_spans,
                    aligned,
                    source_bytes,
                    destination_bytes,
                    "deferred logical fragment cannot be exchanged directly"
                );
                return Ok(None);
            }
            let mut source_index = 0usize;
            let mut destination_index = 0usize;
            let mut source_offset = 0u32;
            let mut destination_offset = 0u32;
            while source_index < source_spans.len() && destination_index < destination_spans.len() {
                let source_remaining = source_spans[source_index].bytes - source_offset;
                let destination_remaining =
                    destination_spans[destination_index].bytes - destination_offset;
                let bytes = source_remaining
                    .min(destination_remaining)
                    .min(maximum_bytes);
                if bytes == 0 || bytes & 0b11 != 0 {
                    return Ok(None);
                }
                fragments = fragments.saturating_add(1);
                source_offset += bytes;
                destination_offset += bytes;
                if source_offset == source_spans[source_index].bytes {
                    source_index += 1;
                    source_offset = 0;
                }
                if destination_offset == destination_spans[destination_index].bytes {
                    destination_index += 1;
                    destination_offset = 0;
                }
            }
            if source_index != source_spans.len()
                || destination_index != destination_spans.len()
                || source_offset != 0
                || destination_offset != 0
            {
                return Ok(None);
            }
        }
        Ok(Some(fragments))
    }

    pub(super) fn mappings_benefit_from_word_exchange(
        &self,
        mappings: &[(ShardView, ShardView)],
        destination: LowShardId,
    ) -> LowLoweringResult<bool> {
        let Some(fragments) = self.mapping_word_exchange_fragments(mappings)? else {
            return Ok(false);
        };
        let shard = &self.shards[destination.index() as usize];
        let bytes = u64::from(crate::shard_storage_bytes(shard)?);
        let elements = bytes.div_ceil(shard.tensor_type.format.precision.bytes().max(1));
        let packed_cycles = crate::estimate::row_major_pack_cycles(&shard.tensor_type, elements);
        let clear_cycles = if self.shard_has_padding(destination) {
            crate::estimate::IPU21_TARGET_COSTS
                .kernel_launch_cycles
                .saturating_add(bytes.div_ceil(8 * 6))
        } else {
            0
        };
        let fragment_cycles = fragments
            .saturating_mul(crate::estimate::IPU21_LOGICAL_FRAGMENT_CYCLES)
            .saturating_add(clear_cycles);
        let direct = fragment_cycles < packed_cycles;
        tracing::trace!(
            destination = destination.index(),
            fragments,
            fragment_cycles,
            packed_cycles,
            direct,
            "selected logical conversion materialization"
        );
        Ok(direct)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn deferred_panel_benefits_from_word_exchange(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
    ) -> LowLoweringResult<bool> {
        let mappings = self.deferred_panel_mappings(
            value,
            stream,
            row_start,
            rows,
            column_start,
            columns,
            destination,
        )?;
        self.mappings_benefit_from_word_exchange(&mappings, destination)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather_deferred_panel(
        &self,
        value: MidValueId,
        stream: u32,
        row_start: u32,
        rows: u32,
        column_start: u32,
        columns: u32,
        destination: LowShardId,
        order: ExchangeOrder,
        transfers: &mut BTreeMap<ShardView, Vec<ShardView>>,
        local_copies: &mut Vec<(u16, LocalCopy)>,
    ) -> LowLoweringResult<()> {
        let destination_tile = self.shards[destination.index() as usize].tile;
        let mappings = self.deferred_panel_mappings(
            value,
            stream,
            row_start,
            rows,
            column_start,
            columns,
            destination,
        )?;
        for (source_view, destination_view) in mappings {
            let mappings = if order == ExchangeOrder::Physical {
                self.f16_micro_panel_mappings(vec![(source_view, destination_view)])?
                    .ok_or(LowLoweringError::InvalidOperatorPlan)?
            } else {
                vec![(source_view, destination_view)]
            };
            for (source_view, destination_view) in mappings {
                if self.shards[source_view.shard.index() as usize].tile == destination_tile {
                    if order == ExchangeOrder::Physical {
                        append_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            destination_tile,
                            local_copies,
                        )?;
                    } else {
                        append_logical_span_copies(
                            &self.shards,
                            &source_view,
                            &destination_view,
                            destination_tile,
                            local_copies,
                        )?;
                    }
                } else {
                    transfers
                        .entry(source_view)
                        .or_default()
                        .push(destination_view);
                }
            }
        }
        Ok(())
    }
}
