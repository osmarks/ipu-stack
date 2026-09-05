//! GEMM tile phases and operand staging.

use super::*;

pub(super) fn split_gemm_matrices(
    run: &KernelRun,
    axis: usize,
    coordinates: &mut [u32],
    runs: &mut Vec<KernelRun>,
) -> LowLoweringResult<()> {
    if axis < coordinates.len() {
        let extent = run
            .output
            .extents
            .get(axis)
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        if extent.logical_end != extent.physical_end {
            return Err(LowLoweringError::InvalidOperatorPlan);
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
) -> LowLoweringResult<()> {
    let input_axes = view.extents.len().saturating_sub(2);
    if input_axes > output_coordinates.len() {
        return Err(LowLoweringError::InvalidOperatorPlan);
    }
    let output_axis_offset = output_coordinates.len() - input_axes;
    for (axis, extent) in view.extents[..input_axes].iter_mut().enumerate() {
        if extent.physical_end - extent.start == 1 {
            continue;
        }
        let coordinate = output_coordinates[output_axis_offset + axis];
        if coordinate < extent.start || coordinate >= extent.physical_end {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        extent.start = coordinate;
        extent.logical_end = coordinate + 1;
        extent.physical_end = coordinate + 1;
    }
    Ok(())
}

impl LoweringState {
    pub(super) fn right_shard_for_block(
        &self,
        right_shards: &[LowShardId],
        tile: u16,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> Option<LowShardId> {
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
        right_shards: &'a [LowShardId],
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = LowShardId> + 'a {
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
        shards: &'a [LowShardId],
        column_axis: usize,
        inner_axis: usize,
        column_start: u32,
        column_end: u32,
        inner_start: u32,
        inner_end: u32,
    ) -> impl Iterator<Item = LowShardId> + 'a {
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
        shards: &[LowShardId],
        tile: u16,
    ) -> Option<LowShardId> {
        shards
            .iter()
            .copied()
            .min_by_key(|shard| u8::from(self.shards[shard.index() as usize].tile != tile))
    }

    pub(super) fn use_uniform_interleaved_gemm_staging(
        &self,
        output_shards: &[LowShardId],
        right_shards: &[LowShardId],
        columns: Range<u32>,
        inner: Range<u32>,
        access_tail: u32,
    ) -> LowLoweringResult<bool> {
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
                    .ok_or(LowLoweringError::IdOverflow)
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
    pub(super) fn lower_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        orientation: crate::GemmOrientation,
        distribution: GemmDistribution,
        requirements: &StorageRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        if let GemmDistribution::ParallelReduction {
            row_partitions,
            column_partitions,
            inner_partitions,
            result_row_partitions,
            result_column_partitions,
            reduction_staging,
        } = distribution
        {
            return self.lower_parallel_reduction_gemm(
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
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if requirements.inputs[1]
            .format
            .layout
            .tiling
            .axes
            .iter()
            .any(|axis| axis.axis == crate::TensorAxis::FromEnd(2) && axis.partitions > 1)
        {
            return self.lower_streamed_blocked_gemm(
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
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
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
            return Err(LowLoweringError::InvalidOperatorPlan);
        }

        let panels_per_phase = column_extent / output_column_block;
        let phase_column_width = output_column_block
            .checked_mul(panels_per_phase)
            .ok_or(LowLoweringError::IdOverflow)?;
        let mut local_right_staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let mut remote_right_staging = vec![
            vec![
                None;
                usize::try_from(panels_per_phase)
                    .map_err(|_| LowLoweringError::IdOverflow)?
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
                        return Err(LowLoweringError::InvalidOperatorPlan);
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
                        return Err(LowLoweringError::InvalidOperatorPlan);
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
                            .map_err(|_| LowLoweringError::IdOverflow)?;
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
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
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
                                let copy = self.push_shard(LowShard {
                                    id: LowShardId(0),
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
                                        .ok_or(LowLoweringError::IdOverflow)?;
                                }
                                self.full_view(copy)
                            }
                        } else {
                            let slot = &mut remote_right_staging[usize::from(tile)][staging_slot];
                            let copy = if let Some(copy) = *slot {
                                copy
                            } else {
                                let copy = self.push_shard(LowShard {
                                    id: LowShardId(0),
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
                            KernelRun::new(
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
                                requirements.clone(),
                            ),
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_parallel_reduction_gemm(
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
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [semantic_left_value, semantic_right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
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
            return Err(LowLoweringError::InvalidOperatorPlan);
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
            return Err(LowLoweringError::InvalidOperatorPlan);
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
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let column_tiling = output_type
            .format
            .layout
            .tiling
            .axes
            .iter()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
        let column_blocks = physical_columns / column_grain;
        if !physical_columns.is_multiple_of(column_grain)
            || column_blocks < u32::from(column_partitions)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let short_blocks = column_blocks / u32::from(column_partitions);
        let long_partitions = column_blocks % u32::from(column_partitions);
        let columns = (0..u32::from(column_partitions))
            .map(|partition| {
                if column_tiling.partitions == column_partitions {
                    return column_tiling
                        .shard_bounds(physical_columns, logical_columns, partition)
                        .map_err(LowLoweringError::from);
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
            .collect::<LowLoweringResult<Vec<_>>>()?;
        let partial_type = operation
            .operator_plan()
            .ok_or(LowLoweringError::InvalidOperatorPlan)?
            .dispatch
            .gemm_partial_tensor(&output_type);

        let mut replica_groups = BTreeMap::<Vec<(u32, u32)>, Vec<LowShardId>>::new();
        for left in left_shards.iter().copied() {
            let key = self.shards[left.index() as usize]
                .extents
                .iter()
                .map(|extent| (extent.start, extent.physical_end))
                .collect::<Vec<_>>();
            replica_groups.entry(key).or_default().push(left);
        }
        let mut replica_columns = BTreeMap::<LowShardId, u16>::new();
        for replicas in replica_groups.values_mut() {
            replicas.sort_unstable_by_key(|shard| self.shards[shard.index() as usize].tile);
            if replicas.len() != usize::from(column_partitions) {
                return Err(LowLoweringError::InvalidOperatorPlan);
            }
            for (column, shard) in replicas.iter().copied().enumerate() {
                replica_columns.insert(
                    shard,
                    u16::try_from(column).map_err(|_| LowLoweringError::IdOverflow)?,
                );
            }
        }
        {
            let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
            let mut local_copies = Vec::<(u16, LocalCopy)>::new();
            let mut gemm_runs = Vec::<(u16, KernelRun)>::new();
            let mut partials = BTreeMap::<Vec<(u32, u32)>, Vec<(u16, ShardView)>>::new();
            let mut resident_lefts = BTreeMap::<LowShardId, ShardView>::new();
            let mut weight_staging = BTreeMap::<(u16, LowShardId), LowShardId>::new();
            for (output_column, &(column_start, logical_column_end, column_end)) in
                columns.iter().enumerate()
            {
                let output_column =
                    u32::try_from(output_column).map_err(|_| LowLoweringError::IdOverflow)?;
                let local_output_columns = column_end - column_start;
                if local_output_columns == 0
                    || local_output_columns > output_column_block
                    || !local_output_columns.is_multiple_of(crate::mid::AMP_COLUMN_MICRO)
                {
                    return Err(LowLoweringError::InvalidOperatorPlan);
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
                            return Err(LowLoweringError::InvalidOperatorPlan);
                        }
                        resident_lefts.insert(left, view.clone());
                        view
                    };
                    let inner = left_shard.extents[left_inner_axis];
                    if !(inner.physical_end - inner.start).is_multiple_of(inner_block) {
                        return Err(LowLoweringError::InvalidOperatorPlan);
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
                                u16::try_from(axis).map_err(|_| LowLoweringError::IdOverflow)?;
                        }
                    }
                    extents[output_row_axis] = left_shard.extents[left_row_axis];
                    extents[output_row_axis].axis =
                        u16::try_from(output_row_axis).map_err(|_| LowLoweringError::IdOverflow)?;
                    extents[output_column_axis] = ShardExtent {
                        axis: u16::try_from(output_column_axis)
                            .map_err(|_| LowLoweringError::IdOverflow)?,
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
                        let partial = self.push_shard(LowShard {
                            id: LowShardId(0),
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
                        .ok_or(LowLoweringError::InvalidOperatorPlan)?;
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
                            let staging = self.push_shard(LowShard {
                                id: LowShardId(0),
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
                                .ok_or(LowLoweringError::InvalidOperatorPlan)?;
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
                                    weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
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
                                        .map_err(|_| LowLoweringError::IdOverflow)?
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
                                        weights.ok_or(LowLoweringError::InvalidOperatorPlan)?
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
                                        weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
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
                                weights.ok_or(LowLoweringError::InvalidOperatorPlan)?
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
                                weights.ok_or(LowLoweringError::InvalidOperatorPlan)?,
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

            let remote_partials_per_stage = match reduction_staging {
                crate::ReductionStaging::Complete => inner_partitions.saturating_sub(1),
                crate::ReductionStaging::Streamed => 1,
            };
            let reduction_stages = inner_partitions
                .saturating_sub(1)
                .div_ceil(remote_partials_per_stage.max(1));
            let mut reduction_transfers = (0..reduction_stages)
                .map(|_| BTreeMap::<ShardView, Vec<ShardView>>::new())
                .collect::<Vec<_>>();
            let mut seed_copies = Vec::<(u16, LocalCopy)>::new();
            let mut reduction_runs = (0..reduction_stages)
                .map(|_| Vec::<(u16, KernelRun)>::new())
                .collect::<Vec<_>>();
            let mut result_copies = Vec::<(u16, LocalCopy)>::new();
            let mut reduction_roots = 0usize;
            for contributors in partials.into_values() {
                let Some((_, complete)) = contributors.first() else {
                    return Err(LowLoweringError::InvalidOperatorPlan);
                };
                let expected = complete
                    .extents
                    .iter()
                    .try_fold(1u64, |elements, extent| {
                        elements.checked_mul(u64::from(extent.physical_end - extent.start))
                    })
                    .ok_or(LowLoweringError::IdOverflow)?;
                let mut covered = 0u64;
                for output in output_shards.iter().copied() {
                    let owner = self.shards[output.index() as usize].clone();
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
                        .ok_or(LowLoweringError::IdOverflow)?;
                    if elements == 0 || !elements.is_multiple_of(8) {
                        return Err(LowLoweringError::InvalidOperatorPlan);
                    }
                    covered = covered
                        .checked_add(u64::from(elements))
                        .ok_or(LowLoweringError::IdOverflow)?;

                    let initial = self.push_packed_buffer(
                        owner.tile,
                        elements,
                        Precision::F16,
                        ShardDefinition::Staging,
                    )?;
                    let remote_elements = elements
                        .checked_mul(u32::from(remote_partials_per_stage))
                        .ok_or(LowLoweringError::IdOverflow)?;
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
                        .position(|(tile, _)| *tile == owner.tile)
                        .unwrap_or(0);
                    let source_view = |partial: &ShardView| ShardView {
                        shard: partial.shard,
                        extents: intersection.clone(),
                    };
                    let seed_source = source_view(&contributors[seed].1);
                    if contributors[seed].0 == owner.tile {
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
                        .map(|(_, (_, partial))| partial)
                        .collect::<Vec<_>>();
                    for (stage, chunk) in remote_contributors
                        .chunks(usize::from(remote_partials_per_stage))
                        .enumerate()
                    {
                        for (slot, partial) in chunk.iter().enumerate() {
                            let start = u32::try_from(slot)
                                .map_err(|_| LowLoweringError::IdOverflow)?
                                .checked_mul(elements)
                                .ok_or(LowLoweringError::IdOverflow)?;
                            let end = start
                                .checked_add(elements)
                                .ok_or(LowLoweringError::IdOverflow)?;
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
                            KernelRun::new(
                                WorkProvenance {
                                    operation: operation.source,
                                    value: Some(*output_value),
                                    reason: WorkReason::OperatorKernel,
                                },
                                TileKernelSpec::ReductionSum {
                                    partials: u16::try_from(chunk.len() + 1)
                                        .map_err(|_| LowLoweringError::IdOverflow)?,
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
                                requirements.clone(),
                            ),
                        ));
                    }
                    let final_result = if usize::from(reduction_stages).is_multiple_of(2) {
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
                    return Err(LowLoweringError::InvalidOperatorPlan);
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
                        operation: operation.source,
                        value: Some(*output_value),
                        reason: WorkReason::OperatorInputs,
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
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_streamed_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        requirements: &StorageRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        if inner_block == 0 || output_column_block == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
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
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let staging_bytes = inner_block
            .checked_mul(output_column_block)
            .and_then(|elements| {
                elements.checked_mul(requirements.inputs[1].format.precision.bytes() as u32)
            })
            .ok_or(LowLoweringError::IdOverflow)?;
        let use_interleaved_staging = requirements.inputs[1].format.precision
            == crate::Precision::F16
            && output_shards.iter().try_fold(true, |available, output| {
                let tile = self.shards[output.index() as usize].tile;
                Ok::<_, LowLoweringError>(
                    available
                        && self.interleaved_capacity_available(
                            tile,
                            staging_bytes,
                            requirements.inputs[1].access_tail_bytes,
                        )?,
                )
            })?;
        let mut staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let mut local_staging = BTreeMap::<(u16, u32), LowShardId>::new();
        let columns_per_phase = column_extent / output_column_block;
        let column_phase_width = output_column_block
            .checked_mul(columns_per_phase)
            .ok_or(LowLoweringError::IdOverflow)?;

        for inner_start in (0..inner_extent).step_by(inner_block as usize) {
            let inner_end = inner_start + inner_block;
            for phase_column_start in (0..column_extent).step_by(column_phase_width as usize) {
                let phase_column_end = phase_column_start
                    .saturating_add(column_phase_width)
                    .min(column_extent);
                let mut transfers = BTreeMap::<ShardView, Vec<ShardView>>::new();
                let mut local_copies = Vec::<(u16, LocalCopy)>::new();
                let mut runs = Vec::with_capacity(output_shards.len());
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
                        return Err(LowLoweringError::InvalidOperatorPlan);
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
                    for output in column_outputs {
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
                            .ok_or(LowLoweringError::InvalidOperatorPlan)?;
                        let right_rank = self.shards[right.index() as usize].extents.len();
                        let right_view = self.narrow_view(
                            right,
                            &[
                                (right_rank - 2, inner_start, inner_end),
                                (right_rank - 1, column_start, column_end),
                            ],
                        )?;
                        let local = self.shards[right.index() as usize].tile == tile;
                        let local_spans = local
                            .then(|| {
                                view_byte_spans(&self.shards[right.index() as usize], &right_view)
                            })
                            .transpose()?;
                        let resident_view =
                            if local_spans.as_ref().is_some_and(|spans| spans.len() == 1)
                                && (!use_interleaved_staging
                                    || requirements.inputs[1].local_staging
                                        == crate::LocalOperandStaging::Direct)
                            {
                                right_view.clone()
                            } else {
                                let selected_staging = if local {
                                    &mut local_staging
                                } else {
                                    &mut staging
                                };
                                let resident = if let Some(resident) =
                                    selected_staging.get(&(tile, column_start)).copied()
                                {
                                    resident
                                } else {
                                    let mut tensor_type =
                                        self.shards[right.index() as usize].tensor_type.clone();
                                    if use_interleaved_staging {
                                        tensor_type.format.layout.memory_class =
                                            crate::MemoryClass::Ipu21Interleaved;
                                    }
                                    let resident = self.push_shard(LowShard {
                                        id: LowShardId(0),
                                        tile,
                                        tensor_type,
                                        extents: right_view.extents.clone(),
                                        definition: if local {
                                            ShardDefinition::Staging
                                        } else {
                                            ShardDefinition::ExchangeStaging
                                        },
                                    })?;
                                    selected_staging.insert((tile, column_start), resident);
                                    resident
                                };
                                if let Some(spans) = local_spans {
                                    let mut destination_offset = 0u32;
                                    for span in spans {
                                        local_copies.push((
                                            tile,
                                            LocalCopy {
                                                source: right,
                                                source_offset: span.offset,
                                                destination: resident,
                                                destination_offset,
                                                bytes: span.bytes,
                                                pattern: CopyPattern::Contiguous,
                                            },
                                        ));
                                        destination_offset = destination_offset
                                            .checked_add(span.bytes)
                                            .ok_or(LowLoweringError::IdOverflow)?;
                                    }
                                } else {
                                    transfers
                                        .entry(right_view.clone())
                                        .or_default()
                                        .push(self.full_view(resident));
                                }
                                self.full_view(resident)
                            };
                        let output_view = self
                            .narrow_view(output, &[(output_rank - 1, column_start, column_end)])?;
                        let mut kernel = if inner_start == 0 {
                            initialize.clone()
                        } else {
                            accumulate.clone()
                        };
                        if self.shards[resident_view.shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .memory_class
                            == crate::MemoryClass::Ipu21Interleaved
                            && let TileKernelSpec::Gemm { weights, .. } = &mut kernel
                        {
                            *weights = crate::GemmWeightLoad::Interleaved;
                        }
                        runs.push((
                            tile,
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
                                        views: vec![resident_view],
                                    },
                                ],
                                output_view,
                                requirements.clone(),
                            ),
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
