//! GEMM tile phases and operand staging.

use super::*;

pub(super) fn split_gemm_matrices(
    run: &KernelRun,
    axis: usize,
    coordinates: &mut [u32],
    runs: &mut Vec<KernelRun>,
) -> BlockBuildResult<()> {
    if axis < coordinates.len() {
        let extent = run
            .output
            .extents
            .get(axis)
            .ok_or(BlockBuildError::InvalidOperatorPlan)?;
        if extent.logical_end != extent.physical_end {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        for coordinate in extent.start..extent.physical_end {
            coordinates[axis] = coordinate;
            split_gemm_matrices(run, axis + 1, coordinates, runs)?;
        }
        return Ok(());
    }

    let mut matrix = run.clone();
    narrow_gemm_matrix_view(&mut matrix.output, coordinates)?;
    for operand in &mut matrix.inputs {
        for view in &mut operand.views {
            narrow_gemm_matrix_view(view, coordinates)?;
        }
    }
    runs.push(matrix);
    Ok(())
}

pub(super) fn narrow_gemm_matrix_view(
    view: &mut ShardView,
    output_coordinates: &[u32],
) -> BlockBuildResult<()> {
    let input_axes = view.extents.len().saturating_sub(2);
    if input_axes > output_coordinates.len() {
        return Err(BlockBuildError::InvalidOperatorPlan);
    }
    let output_axis_offset = output_coordinates.len() - input_axes;
    for (axis, extent) in view.extents[..input_axes].iter_mut().enumerate() {
        if extent.physical_end - extent.start == 1 {
            continue;
        }
        let coordinate = output_coordinates[output_axis_offset + axis];
        if coordinate < extent.start || coordinate >= extent.physical_end {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        extent.start = coordinate;
        extent.logical_end = coordinate + 1;
        extent.physical_end = coordinate + 1;
    }
    Ok(())
}

impl BlockBuilder {
    pub(super) fn right_shard_for_block(
        &self,
        right_shards: &[BlockValueId],
        tile: u16,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> Option<BlockValueId> {
        self.right_shards_for_block(
            right_shards,
            column_start,
            column_end,
            inner_start,
            inner_end,
        )
        .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != tile))
    }

    pub(super) fn right_shards_for_block<'a>(
        &'a self,
        right_shards: &'a [BlockValueId],
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = BlockValueId> + 'a {
        right_shards.iter().copied().filter(move |shard| {
            let extents = &self.shards[shard.index() as usize].extents;
            let columns = extents[extents.len() - 1];
            let inner = extents[extents.len() - 2];
            columns.start <= column_start
                && columns.physical_end >= column_end
                && inner.start <= inner_start
                && inner.physical_end >= inner_end
        })
    }

    pub(super) fn matrix_shards_for_block<'a>(
        &'a self,
        shards: &'a [BlockValueId],
        column_axis: usize,
        inner_axis: usize,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = BlockValueId> + 'a {
        shards.iter().copied().filter(move |shard| {
            let extents = &self.shards[shard.index() as usize].extents;
            let columns = extents[column_axis];
            let inner = extents[inner_axis];
            columns.start <= column_start
                && columns.physical_end >= column_end
                && inner.start <= inner_start
                && inner.physical_end >= inner_end
        })
    }

    pub(super) fn prefer_local_shard(
        &self,
        shards: &[BlockValueId],
        tile: u16,
    ) -> Option<BlockValueId> {
        shards
            .iter()
            .copied()
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != tile))
    }

    pub(super) fn use_uniform_interleaved_gemm_staging(
        &self,
        output_shards: &[BlockValueId],
        right_shards: &[BlockValueId],
        columns: Range<u32>,
        inner: Range<u32>,
        access_tail: u32,
    ) -> BlockBuildResult<bool> {
        let mut candidates = Vec::with_capacity(output_shards.len());
        for output in output_shards {
            let tile = self.shards[output.index() as usize].tile;
            let Some(right) = self.right_shard_for_block(
                right_shards,
                tile,
                columns.start,
                columns.end,
                inner.start,
                inner.end,
            ) else {
                return Ok(false);
            };
            let shard = &self.shards[right.index() as usize];
            if shard.tile != tile || shard.tensor_type.format.precision != crate::Precision::F16 {
                return Ok(false);
            }
            let rank = shard.extents.len();
            if rank < 2 {
                return Ok(false);
            }
            let view = self.narrow_view(
                right,
                &[
                    (rank - 2, inner.start, inner.end),
                    (rank - 1, columns.start, columns.end),
                ],
            )?;
            let spans = view_byte_spans(shard, &view)?;
            if spans.len() <= 1 {
                return Ok(false);
            }
            let bytes = spans.iter().try_fold(0u32, |total, span| {
                total
                    .checked_add(span.bytes)
                    .ok_or(BlockBuildError::IdOverflow)
            })?;
            candidates.push((tile, bytes));
        }
        for (tile, bytes) in candidates {
            if !self.interleaved_capacity_available(tile, bytes, access_tail)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        orientation: crate::GemmOrientation,
        distribution: GemmDistribution,
        requirements: &StorageRequirements,
        tiles: &mut BlockRegion,
    ) -> BlockBuildResult<()> {
        if let GemmDistribution::ParallelReduction {
            row_partitions,
            column_partitions,
            inner_partitions,
            result_row_partitions,
            result_column_partitions,
            reduction_staging,
        } = distribution
        {
            return self.build_parallel_gemm(
                operation,
                initialize,
                accumulate,
                inner_block,
                output_column_block,
                orientation,
                row_partitions,
                column_partitions,
                inner_partitions,
                result_row_partitions,
                result_column_partitions,
                reduction_staging,
                requirements,
                tiles,
            );
        }
        if orientation != crate::GemmOrientation::Normal {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(BlockBuildError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(BlockBuildError::ResultArity);
        };
        if requirements.inputs[1]
            .format
            .layout
            .tiling
            .axes
            .iter()
            .any(|axis| axis.axis == crate::TensorAxis::FromEnd(2) && axis.partitions > 1)
        {
            return self.build_streamed_blocked_gemm(
                operation,
                initialize,
                accumulate,
                inner_block,
                output_column_block,
                requirements,
                tiles,
            );
        }
        if inner_block == 0 || output_column_block == 0 {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }
        let inner_extent = left_type.format.layout.padded_shape(&left_type.shape)?.0[left_rank - 1];
        let column_extent = output_type
            .format
            .layout
            .padded_shape(&output_type.shape)?
            .0[output_rank - 1];
        if !inner_extent.is_multiple_of(inner_block)
            || !column_extent.is_multiple_of(output_column_block)
        {
            return Err(BlockBuildError::InvalidOperatorPlan);
        }

        let panels_per_phase = column_extent / output_column_block;
        let phase_column_width = output_column_block
            .checked_mul(panels_per_phase)
            .ok_or(BlockBuildError::IdOverflow)?;
        let mut local_right_staging = BTreeMap::<(u16, u32), BlockValueId>::new();
        let mut remote_right_staging = vec![
            vec![
                None;
                usize::try_from(panels_per_phase)
                    .map_err(|_| BlockBuildError::IdOverflow)?
            ];
            usize::from(self.tile_count)
        ];
        for phase_column_start in (0..column_extent).step_by(phase_column_width as usize) {
            let phase_column_end = phase_column_start
                .saturating_add(phase_column_width)
                .min(column_extent);
            for inner_start in (0..inner_extent).step_by(inner_block as usize) {
                let inner_end = inner_start + inner_block;
                let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
                let mut local_copies = Vec::<(u16, LocalCopy)>::new();
                let mut runs = Vec::new();
                let mut left_views = BTreeMap::<u16, ShardView>::new();
                for column_start in
                    (phase_column_start..phase_column_end).step_by(output_column_block as usize)
                {
                    let column_end = column_start + output_column_block;
                    let right_candidates = self
                        .right_shards_for_block(
                            &right_shards,
                            column_start,
                            column_end,
                            inner_start,
                            inner_end,
                        )
                        .collect::<Vec<_>>();
                    if right_candidates.is_empty() {
                        return Err(BlockBuildError::InvalidOperatorPlan);
                    }
                    let column_outputs = output_shards
                        .iter()
                        .copied()
                        .filter(|output| {
                            let extents = &self.shards[output.index() as usize].extents;
                            let columns = extents[extents.len() - 1];
                            columns.start <= column_start && columns.physical_end >= column_end
                        })
                        .collect::<Vec<_>>();
                    if column_outputs.is_empty() {
                        return Err(BlockBuildError::InvalidOperatorPlan);
                    }
                    let use_interleaved_staging = self.use_uniform_interleaved_gemm_staging(
                        &column_outputs,
                        &right_shards,
                        column_start..column_end,
                        0..inner_block,
                        requirements.inputs[1].access_tail_bytes,
                    )?;
                    let staging_slot =
                        usize::try_from((column_start - phase_column_start) / output_column_block)
                            .map_err(|_| BlockBuildError::IdOverflow)?;
                    for output in &column_outputs {
                        let tile = self.shards[output.index() as usize].tile;
                        let left_view = if let Some(view) = left_views.get(&tile) {
                            view.clone()
                        } else {
                            let view = self.dispatch_input_view(
                                *left_value,
                                tile,
                                &[(left_rank - 1, inner_start, inner_end)],
                                &mut transfers,
                                &mut local_copies,
                            )?;
                            left_views.insert(tile, view.clone());
                            view
                        };
                        let right = self
                            .prefer_local_shard(&right_candidates, tile)
                            .ok_or(BlockBuildError::InvalidOperatorPlan)?;
                        let right_rank = self.shards[right.index() as usize].extents.len();
                        let right_view = self.narrow_view(
                            right,
                            &[
                                (right_rank - 2, inner_start, inner_end),
                                (right_rank - 1, column_start, column_end),
                            ],
                        )?;
                        let resident_right = if self.shards[right.index() as usize].tile == tile {
                            let spans =
                                view_byte_spans(&self.shards[right.index() as usize], &right_view)?;
                            if spans.len() == 1
                                && (!use_interleaved_staging
                                    || requirements.inputs[1].local_staging
                                        == crate::LocalOperandStaging::Direct)
                            {
                                right_view
                            } else {
                                let existing_staging =
                                    local_right_staging.get(&(tile, column_start)).copied();
                                let definition = existing_staging
                                    .map(ShardDefinition::Alias)
                                    .unwrap_or(ShardDefinition::LocalCopy(right));
                                let mut tensor_type =
                                    self.shards[right.index() as usize].tensor_type.clone();
                                if use_interleaved_staging {
                                    tensor_type.format.layout.memory_class =
                                        crate::MemoryClass::Ipu21Interleaved;
                                }
                                let copy = self.push_shard(BlockValue {
                                    id: BlockValueId(0),
                                    tile,
                                    tensor_type,
                                    extents: right_view.extents.clone(),
                                    definition,
                                })?;
                                local_right_staging
                                    .entry((tile, column_start))
                                    .or_insert(copy);
                                let mut destination_offset = 0u32;
                                for span in spans {
                                    local_copies.push((
                                        tile,
                                        LocalCopy {
                                            source: right,
                                            source_offset: span.offset,
                                            destination: copy,
                                            destination_offset,
                                            bytes: span.bytes,
                                            pattern: CopyPattern::Contiguous,
                                        },
                                    ));
                                    destination_offset = destination_offset
                                        .checked_add(span.bytes)
                                        .ok_or(BlockBuildError::IdOverflow)?;
                                }
                                self.full_view(copy)
                            }
                        } else {
                            let slot = &mut remote_right_staging[usize::from(tile)][staging_slot];
                            let copy = if let Some(copy) = *slot {
                                copy
                            } else {
                                let copy = self.push_shard(BlockValue {
                                    id: BlockValueId(0),
                                    tile,
                                    tensor_type: self.shards[right.index() as usize]
                                        .tensor_type
                                        .clone(),
                                    extents: right_view.extents.clone(),
                                    definition: ShardDefinition::ExchangeStaging,
                                })?;
                                *slot = Some(copy);
                                copy
                            };
                            transfers
                                .entry(right_view.clone())
                                .or_default()
                                .push(self.full_view(copy));
                            self.full_view(copy)
                        };
                        let output_view = self
                            .narrow_view(*output, &[(output_rank - 1, column_start, column_end)])?;
                        let mut selected_kernel = if inner_start == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if self.shards[resident_right.shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .memory_class
                            == crate::MemoryClass::Ipu21Interleaved
                            && let TileKernelSpec::Gemm { weights, .. } = &mut selected_kernel
                        {
                            *weights = crate::GemmWeightLoad::Interleaved;
                        }
                        runs.push((
                            tile,
                            self.kernel_run(
                                WorkProvenance {
                                    operation: operation.source,
                                    value: Some(*output_value),
                                    reason: WorkReason::OperatorKernel,
                                },
                                selected_kernel,
                                vec![
                                    KernelOperand {
                                        views: vec![left_view],
                                    },
                                    KernelOperand {
                                        views: vec![resident_right],
                                    },
                                ],
                                output_view,
                            )?,
                        ));
                    }
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: (!self.deferred_conversions.contains_key(left_value))
                            .then_some(*right_value),
                        reason: if self.deferred_conversions.contains_key(left_value) {
                            WorkReason::OperatorInputs
                        } else {
                            WorkReason::OperatorInput { input: 1 }
                        },
                    },
                    tiles,
                )?;
                for (tile, copy) in local_copies {
                    self.append_local_copy(tiles, tile, copy)?;
                }
                for (tile, run) in runs {
                    self.append_kernel(tiles, tile, run)?;
                }
            }
        }
        Ok(())
    }
}
