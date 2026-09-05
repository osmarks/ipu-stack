//! Parallel row/column/K GEMM blocks followed by ordinary sum construction.

use super::*;

impl BlockBuilder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_parallel_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        orientation: crate::GemmOrientation,
        row_partitions: u16,
        column_partitions: u16,
        inner_partitions: u16,
        result_row_partitions: u16,
        result_column_partitions: u16,
        reduction_staging: crate::ReductionStaging,
        requirements: &StorageRequirements,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        let [semantic_left_value, semantic_right_value] = operation.inputs.as_slice() else {
            return Err(BlockBuildError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
        };
        if inner_block == 0
            || output_column_block == 0
            || row_partitions == 0
            || column_partitions == 0
            || inner_partitions < 2
            || result_row_partitions == 0
            || result_column_partitions == 0
            || result_row_partitions.saturating_mul(result_column_partitions) > inner_partitions
        {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let (left_index, right_index) = orientation.operand_indices();
        let values = [semantic_left_value, semantic_right_value];
        let (left_value, right_value) = (values[left_index], values[right_index]);
        let (left_requirement, right_requirement) = (
            &requirements.inputs[left_index],
            &requirements.inputs[right_index],
        );
        let mut kernel_requirements = requirements.clone();
        if orientation == crate::GemmOrientation::Swapped {
            kernel_requirements.inputs.swap(0, 1);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_rank = self.shards[left_shards[0].index() as usize].extents.len();
        let right_rank = self.shards[right_shards[0].index() as usize].extents.len();
        let output_rank = self.shards[output_shards[0].index() as usize].extents.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let (left_row_axis, left_inner_axis) = orientation.matrix_axes(left_rank);
        let (right_inner_axis, right_column_axis) = orientation.matrix_axes(right_rank);
        let (output_row_axis, output_column_axis) = orientation.matrix_axes(output_rank);
        let output_type = self.shards[output_shards[0].index() as usize]
            .tensor_type
            .clone();
        let logical_columns = output_type.shape.0[output_column_axis];
        let output_padded = output_type.format.layout.padded_shape(&output_type.shape)?;
        let physical_columns = output_padded.0[output_column_axis];
        let column_grain = output_type
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
            .map(|axis| axis.block_size)
            .filter(|grain| *grain != 0)
            .ok_or(BlockBuildError::InvalidOperatorPlan)?;
        let column_tiling = output_type
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
            .ok_or(BlockBuildError::InvalidOperatorPlan)?;
        let column_blocks = physical_columns / column_grain;
        if !physical_columns.is_multiple_of(column_grain)
            || column_blocks < u32::from(column_partitions)
        {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let short_blocks = column_blocks / u32::from(column_partitions);
        let long_partitions = column_blocks % u32::from(column_partitions);
        let columns = (0..u32::from(column_partitions))
            .map(|partition| {
                if column_tiling.partitions == column_partitions {
                    return column_tiling
                        .shard_bounds(physical_columns, logical_columns, partition)
                        .map_err(BlockBuildError::from);
                }
                let start_blocks = partition
                    .saturating_mul(short_blocks)
                    .saturating_add(partition.min(long_partitions));
                let blocks = short_blocks + u32::from(partition < long_partitions);
                let start = start_blocks.saturating_mul(column_grain);
                let physical_end = start_blocks
                    .saturating_add(blocks)
                    .saturating_mul(column_grain);
                Ok((
                    start,
                    physical_end.min(logical_columns).max(start),
                    physical_end,
                ))
            })
            .collect::<BlockBuildResult<Vec<_>>>()?;
        let partial_type = operation
            .operator_plan()
            .ok_or(BlockBuildError::InvalidOperatorPlan)?
            .dispatch
            .gemm_partial_tensor(&output_type);

        let mut replica_groups = BTreeMap::<Vec<(u32, u32)>, Vec<BlockValueId>>::new();
        for left in left_shards.iter().copied() {
            let key = self.shards[left.index() as usize]
                .extents
                .iter()
                .map(|extent| (extent.start, extent.physical_end))
                .collect::<Vec<_>>();
            replica_groups.entry(key).or_default().push(left);
        }
        let mut replica_columns = BTreeMap::<BlockValueId, u16>::new();
        for replicas in replica_groups.values_mut() {
            replicas.sort_unstable_by_key(|shard| self.shards[shard.index() as usize].tile);
            if replicas.len() != usize::from(column_partitions) {
                return Err(BlockBuildError::InvalidOperatorPlan);
            }
            for (column, shard) in replicas.iter().copied().enumerate() {
                replica_columns.insert(
                    shard,
                    u16::try_from(column).map_err(|_| BlockBuildError::IdOverflow)?,
                );
            }
        }
        {
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut local_copies = Vec::<(u16, LocalCopy)>::new();
            let mut gemm_runs = Vec::<(u16, KernelRun)>::new();
            let mut partials = BTreeMap::<Vec<(u32, u32)>, Vec<(u16, ShardView)>>::new();
            let mut resident_lefts = BTreeMap::<BlockValueId, ShardView>::new();
            let mut weight_staging = BTreeMap::<(u16, BlockValueId), BlockValueId>::new();
            for (output_column, &(column_start, logical_column_end, column_end)) in
                columns.iter().enumerate()
            {
                let output_column =
                    u32::try_from(output_column).map_err(|_| BlockBuildError::IdOverflow)?;
                let local_output_columns = column_end - column_start;
                if local_output_columns == 0
                    || local_output_columns > output_column_block
                    || !local_output_columns.is_multiple_of(crate::mid::AMP_COLUMN_MICRO)
                {
                    return Err(BlockBuildError::InvalidOperatorPlan);
                }
                for left in left_shards.iter().copied() {
                    let left_shard = self.shards[left.index() as usize].clone();
                    let resident_left = if let Some(view) = resident_lefts.get(&left) {
                        view.clone()
                    } else {
                        let restrictions = left_shard
                            .extents
                            .iter()
                            .enumerate()
                            .map(|(axis, extent)| (axis, extent.start, extent.physical_end))
                            .collect::<Vec<_>>();
                        let view = self.dispatch_input_view(
                            *left_value,
                            left_shard.tile,
                            &restrictions,
                            &mut transfers,
                            &mut local_copies,
                        )?;
                        if left_requirement.materialization
                            != crate::OperandMaterialization::DispatchSlices
                            && view.shard != left
                        {
                            return Err(BlockBuildError::InvalidOperatorPlan);
                        }
                        resident_lefts.insert(left, view.clone());
                        view
                    };
                    let inner = left_shard.extents[left_inner_axis];
                    if !(inner.physical_end - inner.start).is_multiple_of(inner_block) {
                        return Err(BlockBuildError::InvalidOperatorPlan);
                    }
                    if replica_columns.get(&left).copied().map(u32::from) != Some(output_column) {
                        continue;
                    }
                    let padded_output = partial_type
                        .format
                        .layout
                        .padded_shape(&partial_type.shape)?;
                    let mut extents = partial_type
                        .shape
                        .0
                        .iter()
                        .zip(&padded_output.0)
                        .enumerate()
                        .map(|(axis, (&logical_end, &physical_end))| ShardExtent {
                            axis: u16::try_from(axis).unwrap_or(u16::MAX),
                            start: 0,
                            logical_end,
                            physical_end,
                        })
                        .collect::<Vec<_>>();
                    if orientation == crate::GemmOrientation::Normal {
                        for (axis, extent) in extents
                            .iter_mut()
                            .enumerate()
                            .take(output_rank.saturating_sub(2))
                        {
                            *extent = left_shard.extents[axis];
                            extent.axis =
                                u16::try_from(axis).map_err(|_| BlockBuildError::IdOverflow)?;
                        }
                    }
                    extents[output_row_axis] = left_shard.extents[left_row_axis];
                    extents[output_row_axis].axis =
                        u16::try_from(output_row_axis).map_err(|_| BlockBuildError::IdOverflow)?;
                    extents[output_column_axis] = ShardExtent {
                        axis: u16::try_from(output_column_axis)
                            .map_err(|_| BlockBuildError::IdOverflow)?,
                        start: column_start,
                        logical_end: logical_column_end,
                        physical_end: column_end,
                    };
                    let partial_key = extents
                        .iter()
                        .map(|extent| (extent.start, extent.physical_end))
                        .collect::<Vec<_>>();
                    let direct_output = output_shards.iter().copied().find(|output| {
                        let shard = &self.shards[output.index() as usize];
                        shard.tile == left_shard.tile
                            && shard.tensor_type.format.layout.order
                                == partial_type.format.layout.order
                            && shard.tensor_type.format.layout.memory_class
                                == partial_type.format.layout.memory_class
                            && shard.extents.iter().zip(&extents).all(|(owner, partial)| {
                                owner.start <= partial.start
                                    && owner.logical_end >= partial.logical_end
                                    && owner.physical_end >= partial.physical_end
                            })
                    });
                    let partial = if let Some(output) = direct_output {
                        ShardView {
                            shard: output,
                            extents: extents.clone(),
                        }
                    } else {
                        let partial = self.push_shard(BlockValue {
                            id: BlockValueId(0),
                            tile: left_shard.tile,
                            tensor_type: partial_type.clone(),
                            extents,
                            definition: ShardDefinition::Staging,
                        })?;
                        self.full_view(partial)
                    };
                    partials
                        .entry(partial_key)
                        .or_default()
                        .push((left_shard.tile, partial.clone()));

                    let source_panel_block = right_requirement.format.layout.order;
                    let source_panel_block = match source_panel_block {
                        ElementOrder::BlockMajor(
                            BlockMajorOrder::Matrix {
                                row_block: block, ..
                            }
                            | BlockMajorOrder::TransposedMatrix {
                                row_block: block, ..
                            },
                        ) => u32::from(block),
                        _ => AMP_INNER_BLOCK,
                    };
                    let first_panel_end = inner
                        .start
                        .saturating_add(source_panel_block)
                        .min(inner.physical_end);
                    let first_source = self
                        .matrix_shards_for_block(
                            &right_shards,
                            right_column_axis,
                            right_inner_axis,
                            column_start,
                            column_end,
                            inner.start,
                            first_panel_end,
                        )
                        .next()
                        .ok_or(BlockBuildError::InvalidOperatorPlan)?;
                    let mut weight_type = self.shards[first_source.index() as usize]
                        .tensor_type
                        .clone();
                    weight_type.format.layout.memory_class = crate::MemoryClass::Ipu21Interleaved;
                    let mut weight_extents =
                        self.shards[first_source.index() as usize].extents.clone();
                    weight_extents[right_inner_axis].start = inner.start;
                    weight_extents[right_inner_axis].logical_end = inner.logical_end;
                    weight_extents[right_inner_axis].physical_end = inner.physical_end;
                    let source_inner =
                        self.shards[first_source.index() as usize].extents[right_inner_axis];
                    let source_covers_compute_inner = source_inner.start <= inner.start
                        && source_inner.physical_end >= inner.physical_end;
                    let stage_local_sources =
                        right_requirement.local_staging == crate::LocalOperandStaging::MatchRemote;
                    let weights = if self.shards[first_source.index() as usize].tile
                        == left_shard.tile
                        && source_covers_compute_inner
                        && !stage_local_sources
                    {
                        None
                    } else {
                        let key = (left_shard.tile, first_source);
                        if let Some(staging) = weight_staging.get(&key).copied() {
                            Some(staging)
                        } else {
                            let staging = self.push_shard(BlockValue {
                                id: BlockValueId(0),
                                tile: left_shard.tile,
                                tensor_type: weight_type,
                                extents: weight_extents,
                                definition: ShardDefinition::ExchangeStaging,
                            })?;
                            weight_staging.insert(key, staging);
                            Some(staging)
                        }
                    };

                    for (block_index, inner_start) in (inner.start..inner.physical_end)
                        .step_by(inner_block as usize)
                        .enumerate()
                    {
                        let inner_end = inner_start + inner_block;
                        let mut sources = Vec::new();
                        for panel_start in
                            (inner_start..inner_end).step_by(source_panel_block as usize)
                        {
                            let panel_end = panel_start + source_panel_block;
                            let source = self
                                .matrix_shards_for_block(
                                    &right_shards,
                                    right_column_axis,
                                    right_inner_axis,
                                    column_start,
                                    column_end,
                                    panel_start,
                                    panel_end,
                                )
                                .next()
                                .ok_or(BlockBuildError::InvalidOperatorPlan)?;
                            let target_view = self.narrow_view(
                                source,
                                &[
                                    (right_inner_axis, panel_start, panel_end),
                                    (right_column_axis, column_start, column_end),
                                ],
                            )?;
                            let source_is_local =
                                self.shards[source.index() as usize].tile == left_shard.tile;
                            let consume_direct = source_is_local && !stage_local_sources;
                            if !consume_direct {
                                let destination_view = self.narrow_view(
                                    weights.ok_or(BlockBuildError::InvalidOperatorPlan)?,
                                    &[
                                        (right_inner_axis, panel_start, panel_end),
                                        (right_column_axis, column_start, column_end),
                                    ],
                                )?;
                                if source_is_local {
                                    append_span_copies(
                                        &self.shards,
                                        &target_view,
                                        &destination_view,
                                        left_shard.tile,
                                        &mut local_copies,
                                        CopyOrder::Semantic,
                                    )?;
                                } else {
                                    transfers
                                        .entry(target_view.clone())
                                        .or_default()
                                        .push(destination_view);
                                }
                            }
                            sources.push((target_view, consume_direct));
                        }

                        if sources.len() > 1 && sources.iter().any(|(_, local)| *local) {
                            for (panel_index, (source_view, local)) in
                                sources.into_iter().enumerate()
                            {
                                let panel_start = inner_start
                                    + u32::try_from(panel_index)
                                        .map_err(|_| BlockBuildError::IdOverflow)?
                                        * source_panel_block;
                                let panel_end = panel_start + source_panel_block;
                                let left_view = self.narrow_view(
                                    resident_left.shard,
                                    &[(left_inner_axis, panel_start, panel_end)],
                                )?;
                                let mut kernel = if block_index == 0 && panel_index == 0 {
                                    initialize.clone()
                                } else {
                                    accumulate.clone()
                                };
                                if let TileKernelSpec::Gemm {
                                    weights: load,
                                    inner_block: kernel_inner_block,
                                    output_columns: kernel_output_columns,
                                    ..
                                } = &mut kernel
                                {
                                    *kernel_inner_block = source_panel_block;
                                    *kernel_output_columns = local_output_columns;
                                    let selected = if local {
                                        source_view.shard
                                    } else {
                                        weights.ok_or(BlockBuildError::InvalidOperatorPlan)?
                                    };
                                    *load = if self.shards[selected.index() as usize]
                                        .tensor_type
                                        .format
                                        .layout
                                        .memory_class
                                        == crate::MemoryClass::Ipu21Standard
                                    {
                                        crate::GemmWeightLoad::Standard
                                    } else {
                                        crate::GemmWeightLoad::Interleaved
                                    };
                                }
                                let weight_view = if local {
                                    source_view
                                } else {
                                    self.narrow_view(
                                        weights.ok_or(BlockBuildError::InvalidOperatorPlan)?,
                                        &[
                                            (right_inner_axis, panel_start, panel_end),
                                            (right_column_axis, column_start, column_end),
                                        ],
                                    )?
                                };
                                gemm_runs.push((
                                    left_shard.tile,
                                    KernelRun::new(
                                        WorkProvenance {
                                            operation: operation.source,
                                            value: Some(*output_value),
                                            reason: WorkReason::OperatorKernel,
                                        },
                                        kernel,
                                        vec![
                                            KernelOperand {
                                                views: vec![left_view],
                                            },
                                            KernelOperand {
                                                views: vec![weight_view],
                                            },
                                        ],
                                        partial.clone(),
                                        kernel_requirements.clone(),
                                    ),
                                ));
                            }
                            continue;
                        }

                        let left_view = self.narrow_view(
                            resident_left.shard,
                            &[(left_inner_axis, inner_start, inner_end)],
                        )?;
                        let mut kernel = if block_index == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if let TileKernelSpec::Gemm {
                            weights: load,
                            inner_block: kernel_inner_block,
                            output_columns: kernel_output_columns,
                            ..
                        } = &mut kernel
                        {
                            *kernel_inner_block = inner_block;
                            *kernel_output_columns = local_output_columns;
                            let selected = if sources.len() == 1 && sources[0].1 {
                                sources[0].0.shard
                            } else {
                                weights.ok_or(BlockBuildError::InvalidOperatorPlan)?
                            };
                            *load = if self.shards[selected.index() as usize]
                                .tensor_type
                                .format
                                .layout
                                .memory_class
                                == crate::MemoryClass::Ipu21Standard
                            {
                                crate::GemmWeightLoad::Standard
                            } else {
                                crate::GemmWeightLoad::Interleaved
                            };
                        }
                        let weight_view = if sources.len() == 1 && sources[0].1 {
                            sources.pop().expect("one source").0
                        } else {
                            self.narrow_view(
                                weights.ok_or(BlockBuildError::InvalidOperatorPlan)?,
                                &[
                                    (right_inner_axis, inner_start, inner_end),
                                    (right_column_axis, column_start, column_end),
                                ],
                            )?
                        };
                        let run = KernelRun::new(
                            WorkProvenance {
                                operation: operation.source,
                                value: Some(*output_value),
                                reason: WorkReason::OperatorKernel,
                            },
                            kernel,
                            vec![
                                KernelOperand {
                                    views: vec![left_view],
                                },
                                KernelOperand {
                                    views: vec![weight_view],
                                },
                            ],
                            partial.clone(),
                            kernel_requirements.clone(),
                        );
                        gemm_runs.push((left_shard.tile, run));
                    }
                }
            }
            self.append_phase(
                transfers,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*right_value),
                    reason: WorkReason::OperatorInput {
                        input: orientation.operand_indices().1 as u16,
                    },
                },
                tiles,
            )?;
            for (tile, copy) in local_copies {
                self.append_local_copy(tiles, tile, copy)?;
            }
            for (tile, run) in gemm_runs {
                self.append_kernel(tiles, tile, run)?;
            }

            tracing::debug!(
                partial_groups = partials.len(),
                output_shards = output_shards.len(),
                result_row_partitions,
                result_column_partitions,
                "prepared parallel GEMM partials"
            );

            self.append_sum_partials(
                partials
                    .into_values()
                    .map(|contributors| contributors.into_iter().map(|(_, view)| view).collect()),
                &output_shards,
                reduction_staging,
                requirements,
                WorkProvenance {
                    operation: operation.source,
                    value: Some(*output_value),
                    reason: WorkReason::OperatorKernel,
                },
                tiles,
            )?;
        }
        Ok(())
    }
}
