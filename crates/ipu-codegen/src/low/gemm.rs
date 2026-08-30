use super::*;

struct GemmLowering {
    left_value: MidValueId,
    right_value: MidValueId,
    output_value: MidValueId,
    left_shards: Vec<LowShardId>,
    right_shards: Vec<LowShardId>,
    output_shards: Vec<LowShardId>,
    left_rank: usize,
    right_rank: usize,
    output_rank: usize,
    block: crate::GemmBlockShape,
    initialize: TileKernelSpec,
    accumulate: TileKernelSpec,
}

impl GemmLowering {
    fn bind(
        state: &LoweringState,
        operation: &MidOperation,
        plan: &crate::BlockedGemmPlan,
    ) -> LowLoweringResult<GemmLowering> {
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(LowLoweringError::ResultArity);
        };
        let inner_block = plan.geometry.block.inner;
        let output_column_block = plan.geometry.block.output_columns;
        if inner_block == 0 || output_column_block == 0 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_shards = state.value_shards(*left_value)?.to_vec();
        let right_shards = state.value_shards(*right_value)?.to_vec();
        let output_shards = state.value_shards(*output_value)?.to_vec();
        let left_rank = state.shards[left_shards[0].index() as usize].extents.len();
        let right_rank = state.shards[right_shards[0].index() as usize].extents.len();
        let output_rank = state.shards[output_shards[0].index() as usize]
            .extents
            .len();
        if left_rank < 2 || right_rank < 2 || output_rank < 2 {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        Ok(GemmLowering {
            left_value: *left_value,
            right_value: *right_value,
            output_value: *output_value,
            left_shards,
            right_shards,
            output_shards,
            left_rank,
            right_rank,
            output_rank,
            block: plan.geometry.block,
            initialize: gemm_kernel_spec(
                plan.kernel,
                GemmKernelMode::Initialize,
                plan.geometry.block,
            ),
            accumulate: gemm_kernel_spec(
                plan.kernel,
                GemmKernelMode::Accumulate,
                plan.geometry.block,
            ),
        })
    }
}

impl LoweringState {
    pub(super) fn lower_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        plan: &crate::BlockedGemmPlan,
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let gemm = GemmLowering::bind(self, operation, plan)?;
        if matches!(
            plan.geometry.distribution,
            GemmDistribution::ParallelReduction(_)
        ) {
            return self.lower_parallel_reduction_gemm(operation, plan, requirements, gemm, tiles);
        }
        if plan.geometry.orientation != crate::GemmOrientation::Normal {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let left_type = &self.shards[gemm.left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[gemm.output_shards[0].index() as usize].tensor_type;
        let inner_extent = left_type
            .format
            .layout
            .resolve(&left_type.shape)?
            .padded_shape()
            .0[gemm.left_rank - 1];
        let column_extent = output_type
            .format
            .layout
            .resolve(&output_type.shape)?
            .padded_shape()
            .0[gemm.output_rank - 1];
        if !inner_extent.is_multiple_of(gemm.block.inner)
            || !column_extent.is_multiple_of(gemm.block.output_columns)
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
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
                gemm,
                [inner_extent, column_extent],
                requirements,
                tiles,
            );
        }
        let (left_value, right_value, output_value) =
            (&gemm.left_value, &gemm.right_value, &gemm.output_value);
        let (right_shards, output_shards) = (&gemm.right_shards, &gemm.output_shards);
        let (left_rank, output_rank) = (gemm.left_rank, gemm.output_rank);
        let (inner_block, output_column_block) = (gemm.block.inner, gemm.block.output_columns);
        let (initialize, accumulate) = (&gemm.initialize, &gemm.accumulate);

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
                        requirements.inputs[1].allocation.access_tail_bytes,
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
                                        crate::MemoryClass::Interleaved;
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
                                            pattern: LocalCopyPattern::Contiguous,
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
                            == crate::MemoryClass::Interleaved
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
                                KernelRequirements::Operator(requirements.clone()),
                            ),
                        ));
                    }
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: (!self.deferred_values.contains_key(left_value))
                            .then_some(*right_value),
                        reason: if self.deferred_values.contains_key(left_value) {
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

    fn lower_parallel_reduction_gemm(
        &mut self,
        operation: &MidOperation,
        plan: &crate::BlockedGemmPlan,
        requirements: &OperatorRequirements,
        gemm: GemmLowering,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let GemmDistribution::ParallelReduction(reduction) = plan.geometry.distribution else {
            return Err(LowLoweringError::InvalidOperatorPlan);
        };
        let (inner_block, output_column_block) = (gemm.block.inner, gemm.block.output_columns);
        let orientation = plan.geometry.orientation;
        let initialize = gemm.initialize;
        let accumulate = gemm.accumulate;
        let row_partitions = reduction.compute.rows;
        let column_partitions = reduction.compute.columns;
        let inner_partitions = reduction.compute.inner;
        let result_row_partitions = plan
            .geometry
            .result
            .rows
            .checked_div(reduction.compute.rows)
            .unwrap_or(0);
        let result_column_partitions = plan
            .geometry
            .result
            .columns
            .checked_div(reduction.compute.columns)
            .unwrap_or(0);
        let reduction_staging = reduction.staging;
        let semantic_left_value = gemm.left_value;
        let semantic_right_value = gemm.right_value;
        let output_value = gemm.output_value;
        let semantic_left_shards = gemm.left_shards;
        let semantic_right_shards = gemm.right_shards;
        let output_shards = gemm.output_shards;
        let semantic_left_rank = gemm.left_rank;
        let semantic_right_rank = gemm.right_rank;
        let output_rank = gemm.output_rank;
        if row_partitions == 0
            || column_partitions == 0
            || inner_partitions < 2
            || result_row_partitions == 0
            || result_column_partitions == 0
            || result_row_partitions.saturating_mul(result_column_partitions) > inner_partitions
        {
            return Err(LowLoweringError::InvalidOperatorPlan);
        }
        let (
            left_value,
            right_value,
            left_shards,
            right_shards,
            left_rank,
            right_rank,
            left_requirement,
            right_requirement,
        ) = match orientation {
            crate::GemmOrientation::Normal => (
                &semantic_left_value,
                &semantic_right_value,
                semantic_left_shards,
                semantic_right_shards,
                semantic_left_rank,
                semantic_right_rank,
                &requirements.inputs[0],
                &requirements.inputs[1],
            ),
            crate::GemmOrientation::Swapped => (
                &semantic_right_value,
                &semantic_left_value,
                semantic_right_shards,
                semantic_left_shards,
                semantic_right_rank,
                semantic_left_rank,
                &requirements.inputs[1],
                &requirements.inputs[0],
            ),
        };
        let mut kernel_requirements = requirements.clone();
        if orientation == crate::GemmOrientation::Swapped {
            kernel_requirements.inputs.swap(0, 1);
        }
        let output_value = &output_value;
        let (
            left_row_axis,
            left_inner_axis,
            right_inner_axis,
            right_column_axis,
            output_row_axis,
            output_column_axis,
        ) = match orientation {
            crate::GemmOrientation::Normal => (
                left_rank - 2,
                left_rank - 1,
                right_rank - 2,
                right_rank - 1,
                output_rank - 2,
                output_rank - 1,
            ),
            crate::GemmOrientation::Swapped => (
                left_rank - 1,
                left_rank - 2,
                right_rank - 1,
                right_rank - 2,
                output_rank - 1,
                output_rank - 2,
            ),
        };
        let output_type = self.shards[output_shards[0].index() as usize]
            .tensor_type
            .clone();
        let logical_columns = output_type.shape.0[output_column_axis];
        let output_layout = output_type.format.layout.resolve(&output_type.shape)?;
        let physical_columns = output_layout.padded_shape().0[output_column_axis];
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
                    return output_layout
                        .axis_bounds(output_column_axis, partition)
                        .ok_or(LowLoweringError::InvalidOperatorPlan);
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
        let mut partial_type = output_type.clone();
        let partial_tiles = row_partitions.saturating_mul(column_partitions);
        partial_type.format.layout = match (orientation, output_type.format.layout.order) {
            (crate::GemmOrientation::Normal, StorageOrder::Native(NativeKernelOrder::Left)) => {
                Layout::amp_left_result_grid(
                    output_column_block,
                    partial_tiles,
                    row_partitions,
                    column_partitions,
                    crate::operator::GridOrder::ColumnsFast,
                )
            }
            (
                crate::GemmOrientation::Swapped,
                StorageOrder::Native(NativeKernelOrder::TransposedLeft),
            ) => Layout::amp_transposed_left_result_grid(
                output_column_block,
                partial_tiles,
                row_partitions,
                column_partitions,
                crate::operator::GridOrder::ColumnsFast,
            ),
            (crate::GemmOrientation::Normal, _) => Layout::amp_output_grid(
                output_column_block,
                partial_tiles,
                row_partitions,
                column_partitions,
                crate::operator::GridOrder::ColumnsFast,
            ),
            (crate::GemmOrientation::Swapped, _) => Layout::amp_transposed_output_grid(
                output_column_block,
                partial_tiles,
                row_partitions,
                column_partitions,
                crate::operator::GridOrder::ColumnsFast,
            ),
        };
        if let Some(axis) = partial_type
            .format
            .layout
            .tiling
            .axes
            .iter_mut()
            .find(|axis| axis.axis.resolve(output_rank).ok() == Some(output_column_axis))
        {
            axis.block_size = column_grain;
            axis.padding_multiple = column_grain;
            if column_tiling.partitions == column_partitions {
                axis.block_size = column_tiling.block_size;
                axis.padding_multiple = column_tiling.padding_multiple;
                axis.shard_padding_multiple = column_tiling.shard_padding_multiple;
            }
        }

        let mut replica_groups = BTreeMap::<TensorRegion, Vec<LowShardId>>::new();
        for left in left_shards.iter().copied() {
            let key = self.shards[left.index() as usize].extents.physical();
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
            let mut partials = BTreeMap::<TensorRegion, Vec<(u16, ShardView)>>::new();
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
                    || !local_output_columns.is_multiple_of(crate::layout::AMP_COLUMN_MICRO)
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
                    let partial_layout = partial_type.format.layout.resolve(&partial_type.shape)?;
                    let padded_output = partial_layout.padded_shape();
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
                        for axis in 0..output_rank.saturating_sub(2) {
                            extents[axis] = left_shard.extents[axis];
                            extents[axis].axis =
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
                    let partial_key = TensorRegion::new(extents.iter().copied()).physical();
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
                            extents: extents.clone().into(),
                        }
                    } else {
                        let partial = self.push_shard(LowShard {
                            id: LowShardId(0),
                            tile: left_shard.tile,
                            tensor_type: partial_type.clone(),
                            extents: extents.into(),
                            definition: ShardDefinition::Staging,
                        })?;
                        self.full_view(partial)
                    };
                    partials
                        .entry(partial_key)
                        .or_default()
                        .push((left_shard.tile, partial.clone()));

                    let source_panel_block = right_requirement.format.layout.order.clone();
                    let source_panel_block = match source_panel_block {
                        StorageOrder::Blocked(order) => u32::from(order.block_shape[0]),
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
                    weight_type.format.layout.memory_class = crate::MemoryClass::Interleaved;
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
                                    append_logical_span_copies(
                                        &self.shards,
                                        &target_view,
                                        &destination_view,
                                        left_shard.tile,
                                        &mut local_copies,
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
                                        == crate::MemoryClass::Standard
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
                                        KernelRequirements::Operator(kernel_requirements.clone()),
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
                                == crate::MemoryClass::Standard
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
                            KernelRequirements::Operator(kernel_requirements.clone()),
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
                        input: match orientation {
                            crate::GemmOrientation::Normal => 1,
                            crate::GemmOrientation::Swapped => 0,
                        },
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
                        extents: intersection.clone().into(),
                    };
                    let seed_source = source_view(&contributors[seed].1);
                    if contributors[seed].0 == owner.tile {
                        append_span_copies(
                            &self.shards,
                            &seed_source,
                            &self.full_view(initial),
                            owner.tile,
                            &mut seed_copies,
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
                                    }]
                                    .into(),
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
                                KernelRequirements::Operator(requirements.clone()),
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
                            extents: intersection.into(),
                        },
                        owner.tile,
                        &mut result_copies,
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

    fn lower_streamed_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        gemm: GemmLowering,
        [inner_extent, column_extent]: [u32; 2],
        requirements: &OperatorRequirements,
        tiles: &mut [TileWorkList],
    ) -> LowLoweringResult<()> {
        let (left_value, right_value, output_value) =
            (&gemm.left_value, &gemm.right_value, &gemm.output_value);
        let (right_shards, output_shards) = (&gemm.right_shards, &gemm.output_shards);
        let (left_rank, output_rank) = (gemm.left_rank, gemm.output_rank);
        let (inner_block, output_column_block) = (gemm.block.inner, gemm.block.output_columns);
        let (initialize, accumulate) = (&gemm.initialize, &gemm.accumulate);
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
                            requirements.inputs[1].allocation.access_tail_bytes,
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
                                            crate::MemoryClass::Interleaved;
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
                                                pattern: LocalCopyPattern::Contiguous,
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
                            == crate::MemoryClass::Interleaved
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
                                KernelRequirements::Operator(requirements.clone()),
                            ),
                        ));
                    }
                }
                self.append_phase(
                    transfers,
                    WorkProvenance {
                        operation: operation.source,
                        value: (!self.deferred_values.contains_key(left_value))
                            .then_some(*right_value),
                        reason: if self.deferred_values.contains_key(left_value) {
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
