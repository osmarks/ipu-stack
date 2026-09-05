//! Bounded streamed GEMM panel construction.

use super::*;

impl TileGraphBuilder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_streamed_blocked_gemm(
        &mut self,
        operation: &MidOperation,
        initialize: TileKernelSpec,
        accumulate: TileKernelSpec,
        inner_block: u32,
        output_column_block: u32,
        requirements: &StorageRequirements,
        tiles: &mut BlockRegion,
    ) -> ExpansionResult<()> {
        let [left_value, right_value] = operation.inputs.as_slice() else {
            return Err(ExpansionError::InvalidOperatorPlan);
        };
        let [output_value] = operation.results.as_slice() else {
            return Err(ExpansionError::ResultArity);
        };
        if inner_block == 0 || output_column_block == 0 {
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let left_shards = self.value_shards(*left_value)?.to_vec();
        let right_shards = self.value_shards(*right_value)?.to_vec();
        let output_shards = self.value_shards(*output_value)?.to_vec();
        let left_type = &self.shards[left_shards[0].index() as usize].tensor_type;
        let output_type = &self.shards[output_shards[0].index() as usize].tensor_type;
        let left_rank = left_type.shape.0.len();
        let output_rank = output_type.shape.0.len();
        if left_rank < 2 || output_rank < 2 {
            return Err(ExpansionError::InvalidOperatorPlan);
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
            return Err(ExpansionError::InvalidOperatorPlan);
        }
        let staging_bytes = inner_block
            .checked_mul(output_column_block)
            .and_then(|elements| {
                elements.checked_mul(requirements.inputs[1].format.precision.bytes() as u32)
            })
            .ok_or(ExpansionError::IdOverflow)?;
        let use_interleaved_staging = requirements.inputs[1].format.precision
            == crate::Precision::F16
            && output_shards.iter().try_fold(true, |available, output| {
                let tile = self.shards[output.index() as usize].tile;
                Ok::<_, ExpansionError>(
                    available
                        && self.interleaved_capacity_available(
                            tile,
                            staging_bytes,
                            requirements.inputs[1].access_tail_bytes,
                        )?,
                )
            })?;
        let mut staging = BTreeMap::<(u16, u32), BlockValueId>::new();
        let mut local_staging = BTreeMap::<(u16, u32), BlockValueId>::new();
        let columns_per_phase = column_extent / output_column_block;
        let column_phase_width = output_column_block
            .checked_mul(columns_per_phase)
            .ok_or(ExpansionError::IdOverflow)?;

        for inner_start in (0..inner_extent).step_by(inner_block as usize) {
            let inner_end = inner_start + inner_block;
            for phase_column_start in (0..column_extent).step_by(column_phase_width as usize) {
                let phase_column_end = phase_column_start
                    .saturating_add(column_phase_width)
                    .min(column_extent);
                let mut batch = MaterializationBatch::default();
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
                        return Err(ExpansionError::InvalidOperatorPlan);
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
                                operation_provenance(operation),
                                &mut batch,
                                tiles,
                            )?;
                            left_views.insert(tile, view.clone());
                            view
                        };
                        let right = self
                            .prefer_local_shard(&right_candidates, tile)
                            .ok_or(ExpansionError::InvalidOperatorPlan)?;
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
                                    let resident = self.push_shard(BlockValue {
                                        id: BlockValueId(0),
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
                                        batch.after.push((
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
                                            .ok_or(ExpansionError::IdOverflow)?;
                                    }
                                } else {
                                    batch
                                        .semantic
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
                            self.kernel_run(
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
                            )?,
                        ));
                    }
                }
                self.append_materialization(
                    batch,
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
                for (tile, run) in runs {
                    self.append_kernel(tiles, tile, run)?;
                }
            }
        }
        Ok(())
    }
}
