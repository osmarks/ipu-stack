//! Distributed attention stages and bounded key-block materialization.

use super::*;

impl Builder {
    pub(super) fn attention(
        &mut self,
        output: &TensorType,
        key_block: u32,
        query_width: u32,
        value_width: u32,
        materialized: bool,
        query_key_grid: Option<ProductGrid>,
        probability_value_grid: Option<ProductGrid>,
        fp8_scales: [Option<i8>; 2],
    ) -> Option<MidValueId> {
        if !materialized && (query_key_grid.is_some() || probability_value_grid.is_some()) {
            return None;
        }
        // Online softmax merging retains F32 state, but model activations use
        // the selected output precision rather than inheriting that scratch type.
        let final_output = output;
        let mut accumulator = output.clone();
        accumulator.format.precision = Precision::F32;
        // Online merging stores the running maximum and denominator after
        // each value row. These are real storage, not disposable AMP padding.
        accumulator.shape.0[2] += 2;
        let output = &accumulator;
        let product = |inner_block, output_columns, axes| Product {
            multiply: Precision::F16,
            accumulate: AccumulationPrecision::F32,
            mode: GemmKernelMode::Initialize,
            inner_block,
            output_columns,
            axes,
            operands: Default::default(),
            output_aliases: Vec::new(),
        };
        let query = self.tensor(MidValueId(0)).clone();
        let key = self.tensor(MidValueId(1)).clone();
        let value = self.tensor(MidValueId(2)).clone();
        let rank = output.shape.0.len();
        if rank != 3 || key_block == 0 {
            return None;
        }
        let key_rows = key.shape.0[1];
        let mut query_type = query.clone();
        query_type.format.layout = output.format.layout.clone();
        query_type.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        query_type.shape.0[2] = query_width;
        let query_buffer = if query_key_grid.is_some() {
            MidValueId(0)
        } else {
            self.copy(MidValueId(0), query_type, vec![])
        };
        let mut scores_type = output.clone();
        scores_type.format.precision = Precision::F16;
        scores_type.shape.0[2] = key_block;
        scores_type.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        scores_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
        let mut weights_type = scores_type.clone();
        if let Some(scale_exponent) = fp8_scales[1] {
            if !materialized || !key_block.is_multiple_of(32) {
                return None;
            }
            weights_type.format.precision = Precision::F8F143 { scale_exponent };
            // FP32 max/sum, segmented statistics and a 16-half masked tail.
            weights_type.shape.0[2] = key_block + 64;
        } else {
            weights_type.shape.0[2] = key_block + AMP_COLUMN_MICRO;
        }
        weights_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
        let mut product_type = scores_type.clone();
        product_type.shape.0[2] = value_width;
        let qk_axes = ProductAxes {
            valid_inner: Some(query.shape.0[2]),
            valid_columns: None,
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(1),
            output_column: TensorAxis::FromEnd(1),
        };
        let pv_axes = ProductAxes {
            valid_inner: None,
            valid_columns: Some(final_output.shape.0[2]),
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(2),
            output_column: TensorAxis::FromEnd(1),
        };
        let mut packed_key = key;
        let mut packed_value = value;
        for (tensor, width) in [
            (&mut packed_key, query_width),
            (&mut packed_value, value_width),
        ] {
            tensor.shape.0[1] = key_rows;
            tensor.shape.0[2] = width;
            tensor.format.layout.tiling = project_grid(output, tensor, 0, 0)?;
            tensor.format.layout.tiling.axes.push(AxisTiling::new(
                TensorAxis::FromStart(1),
                1,
                key_block,
                Padding::Zero,
            ));
            tensor.format.layout.tiling.axes.push(AxisTiling::new(
                TensorAxis::FromStart(2),
                1,
                AMP_COLUMN_MICRO,
                Padding::Zero,
            ));
            tensor.format.layout.memory_class = MemoryClass::Ipu21Standard;
        }
        packed_key.format.layout.order = ElementOrder::Amp(AmpOrder::TransposedRight);
        packed_value.format.layout.order = ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: u16::try_from(if fp8_scales[1].is_some() {
                key_block.min(AMP_INNER_BLOCK)
            } else {
                key_block
            })
            .ok()?,
            column_block: AMP_COLUMN_MICRO as u16,
        });
        let key_panels = self.prepare_attention_operand(MidValueId(1), &packed_key, key_block)?;
        let value_panels =
            self.prepare_attention_operand(MidValueId(2), &packed_value, key_block)?;
        // Quantize each unreplicated native panel once, before PV ownership
        // replicates it across query partitions.
        let value_panels = if let Some(scale_exponent) = fp8_scales[1] {
            self.cast(value_panels, Precision::F8F143 { scale_exponent })
        } else {
            value_panels
        };
        let mut weights = None;
        let mut result = None;
        for start in (0..key_rows).step_by(key_block as usize) {
            let valid = key_block.min(key_rows - start);
            let mut key_block_type = packed_key.clone();
            let mut value_block_type = packed_value.clone();
            key_block_type.shape.0[1] = valid;
            value_block_type.shape.0[1] = valid;
            // Small streaming coefficient blocks benefit from double-width
            // loads. Interleaving a full K/V matrix partitions too much SRAM
            // away from the large standard-memory projection weights.
            if !materialized {
                key_block_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
                value_block_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
            }
            let k = if query_key_grid.is_some() {
                key_panels
            } else {
                self.copy(key_panels, key_block_type, vec![0, start, 0])
            };
            // Flash broadcasts K/V together. Full materialization keeps their
            // large resident matrices in disjoint lifetimes.
            let v = (!materialized)
                .then(|| self.copy(value_panels, value_block_type.clone(), vec![0, start, 0]));
            let scores = if let Some(grid) = query_key_grid {
                // Softmax consumes a padded row but only its valid key prefix.
                // Retain that logical bound so copies may transfer the shared
                // zero padding instead of splitting an odd FP16 tail word.
                let mut rows = scores_type.clone();
                rows.shape.0[2] = valid;
                for axis in &mut rows.format.layout.tiling.axes {
                    if axis.axis.resolve(3).ok()? == 2 {
                        axis.block_size = key_block;
                        axis.padding_multiple = key_block;
                    }
                }
                self.distributed_product(
                    query_buffer,
                    k,
                    &rows,
                    ProductAxes {
                        valid_columns: Some(valid),
                        ..qk_axes
                    },
                    grid,
                    fp8_scales[0],
                )?
            } else {
                self.compute(
                    vec![query_buffer, k],
                    scores_type.clone(),
                    Compute::Product(product(
                        query_width,
                        key_block,
                        ProductAxes {
                            valid_columns: Some(valid),
                            ..qk_axes
                        },
                    )),
                    None,
                )
            };
            weights = Some(self.kernel(
                vec![scores],
                weights_type.clone(),
                TileKernelSpec::AttentionSoftmax {
                    head_dimension: query.shape.0[2],
                    key_columns: valid,
                    padded_key_columns: key_block,
                },
                weights,
                vec![OperandIndexing::local()],
            ));
            let weights_id = weights?;
            let v = if probability_value_grid.is_some() {
                value_panels
            } else {
                v.unwrap_or_else(|| self.copy(value_panels, value_block_type, vec![0, start, 0]))
            };
            let product = if let Some(grid) = probability_value_grid {
                // The softmax buffer also carries FP32 maximum/denominator
                // words after key_block. Expose only probabilities and their
                // zero padding: a distributed K grid can pad beyond key_block,
                // and copying those statistics as FP16 yields NaNs even when
                // the corresponding V coefficients are zero.
                let mut probabilities = self.tensor(weights_id).clone();
                probabilities.shape.0[2] = key_block;
                let probabilities = self.copy(weights_id, probabilities, vec![]);
                self.distributed_product(
                    probabilities,
                    v,
                    &product_type,
                    ProductAxes {
                        valid_inner: Some(valid),
                        ..pv_axes
                    },
                    grid,
                    fp8_scales[1],
                )?
            } else {
                self.compute(
                    vec![weights_id, v],
                    product_type.clone(),
                    Compute::Product(Product {
                        operands: [
                            OperandWindow(vec![(2, 0, key_block)]),
                            OperandWindow::default(),
                        ],
                        ..product(
                            key_block,
                            value_width,
                            ProductAxes {
                                valid_inner: Some(valid),
                                ..pv_axes
                            },
                        )
                    }),
                    None,
                )
            };
            let final_block = materialized || start + key_block >= key_rows;
            let direct_f16 = final_block && final_output.format.precision == Precision::F16;
            let mut inputs = vec![product, weights_id];
            if direct_f16 {
                // An explicit previous accumulator keeps FP32 state live while
                // the final merge writes a separate, compact FP16 result.
                // The initial-block path does not read this operand.
                inputs.push(result.unwrap_or(product));
            }
            let indexing = vec![OperandIndexing::local(); inputs.len()];
            result = Some(self.kernel(
                inputs,
                if direct_f16 {
                    final_output.clone()
                } else {
                    output.clone()
                },
                TileKernelSpec::AttentionMerge {
                    value_dimension: final_output.shape.0[2],
                    padded_value_dimension: value_width,
                    key_block_columns: key_block,
                    initial: start == 0,
                    final_block,
                },
                if direct_f16 { None } else { result },
                indexing,
            ));
        }
        let result = self.cast(result?, final_output.format.precision);
        Some(self.copy(result, final_output.clone(), vec![]))
    }

    /// Pack once on a small distributed owner grid, then broadcast native
    /// panels. Both materializations are ordinary mid values and copies.
    fn prepare_attention_operand(
        &mut self,
        input: MidValueId,
        resident: &TensorType,
        key_block: u32,
    ) -> Option<MidValueId> {
        // K consists of independent 64-row AMP panels even when the consumer
        // uses the entire key matrix. Distribute preparation by panel instead
        // of tying its ownership to the consumer's GEMM block size.
        let preparation_rows = match resident.format.layout.order {
            ElementOrder::Amp(AmpOrder::TransposedRight) => key_block.min(AMP_INNER_BLOCK),
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix { row_block, .. }) => {
                u32::from(row_block)
            }
            _ => key_block,
        };
        // A short query (MAP pooling has one row) may use far fewer tiles than
        // K/V preparation. Keep the input's distributed owner budget rather
        // than concentrating the entire key sequence on the query owners.
        let preparation_tiles = self
            .tensor(input)
            .format
            .layout
            .tiling
            .tile_count
            .max(resident.format.layout.tiling.tile_count);
        let heads = u16::try_from(resident.shape.0[0]).ok()?;
        let blocks = u16::try_from(resident.shape.0[1].div_ceil(preparation_rows))
            .ok()?
            .min(preparation_tiles / heads)
            .max(1);
        let columns = u16::try_from(resident.shape.0[2].div_ceil(AMP_COLUMN_MICRO))
            .ok()?
            .min(preparation_tiles / heads / blocks)
            .max(1);
        let mut packed = resident.clone();
        packed.format.layout.tiling.tile_count = heads.checked_mul(columns)?.checked_mul(blocks)?;
        packed.format.layout.tiling.replicas = 1;
        for axis in &mut packed.format.layout.tiling.axes {
            match axis.axis.resolve(3).ok()? {
                0 => {
                    axis.partitions = heads;
                    axis.tile_stride = Some(1);
                }
                1 => {
                    axis.partitions = blocks;
                    axis.block_size = preparation_rows;
                    axis.tile_stride = Some(heads.checked_mul(columns)?);
                }
                2 => {
                    axis.partitions = columns;
                    axis.tile_stride = Some(heads);
                }
                _ => unreachable!(),
            }
        }
        Some(self.copy(input, packed, vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_pv_excludes_softmax_statistics_from_padding() {
        let tensor = |shape: [u32; 3]| TensorType {
            shape: TensorShape(shape.to_vec()),
            format: TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_output(1, 1),
            },
        };
        let mut builder = Builder::new(&[
            tensor([1, 2, 72]),
            tensor([1, 729, 72]),
            tensor([1, 729, 72]),
        ]);
        let output = builder
            .attention(
                &tensor([1, 2, 72]),
                768,
                80,
                80,
                true,
                Some(ProductGrid {
                    rows: 1,
                    columns: 1,
                    inner: 1,
                }),
                Some(ProductGrid {
                    rows: 1,
                    columns: 1,
                    inner: 5,
                }),
                [None; 2],
            )
            .unwrap();
        builder.program.outputs = vec![output];
        builder.program.tile_count = 5;
        let expanded = crate::expand_tiles(&builder.program).unwrap();
        let softmax = expanded
            .kernel_runs
            .iter()
            .find(|run| matches!(run.kernel, TileKernelSpec::AttentionSoftmax { .. }))
            .unwrap()
            .outputs[0]
            .shard;
        // The five PV groups consume 800 columns. The first 768 are weights;
        // [768, 784) contains FP32 metadata that can encode FP16 NaNs.
        let mut probability_transfers = 0;
        for phase in &expanded.exchange_phases {
            for transfer in &phase.transfers {
                if transfer.source.shard == softmax {
                    assert!(transfer.source.extents[2].physical_end <= 768);
                    probability_transfers += 1;
                }
            }
        }
        assert!(probability_transfers > 0);
    }

    #[test]
    fn single_query_keeps_distributed_key_preparation() {
        let key = TensorType {
            shape: TensorShape(vec![16, 729, 80]),
            format: TensorFormat {
                precision: Precision::F16,
                layout: Layout::attention_key(16, 12),
            },
        };
        let mut resident = key.clone();
        resident.format.layout = Layout::attention_output(16, 1);
        resident.format.layout.order = ElementOrder::Amp(AmpOrder::TransposedRight);
        let mut b = Builder::new(&[key]);
        let prepared = b
            .prepare_attention_operand(MidValueId(0), &resident, 64)
            .unwrap();
        assert_eq!(b.tensor(prepared).format.layout.tiling.tile_count, 192);
    }
}
