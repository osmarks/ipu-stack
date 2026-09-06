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
    ) -> Option<MidValueId> {
        let product = |inner_block, output_columns| TileKernelSpec::Gemm {
            multiply: Precision::F16,
            accumulate: AccumulationPrecision::F32,
            mode: GemmKernelMode::Initialize,
            weights: GemmWeightLoad::Standard,
            inner_block,
            output_columns,
        };
        let query_key = product(query_width, key_block);
        let probability_value = product(key_block, value_width);
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
        let query_buffer = self.copy(MidValueId(0), query_type, vec![]);
        let mut scores_type = output.clone();
        scores_type.format.precision = Precision::F16;
        scores_type.shape.0[2] = key_block;
        scores_type.format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        scores_type.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
        let mut weights_type = scores_type.clone();
        weights_type.shape.0[2] = key_block + AMP_COLUMN_MICRO;
        weights_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
        let mut product_type = scores_type.clone();
        product_type.shape.0[2] = value_width;
        let qk_axes = ProductAxes {
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(1),
            output_column: TensorAxis::FromEnd(1),
        };
        let pv_axes = ProductAxes {
            left_inner: TensorAxis::FromEnd(1),
            right_inner: TensorAxis::FromEnd(2),
            output_column: TensorAxis::FromEnd(1),
        };
        let mut packed_key = key.clone();
        let mut packed_value = value.clone();
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
            row_block: u16::try_from(key_block).ok()?,
            column_block: AMP_COLUMN_MICRO as u16,
        });
        let key_panels = self.prepare_attention_operand(MidValueId(1), &packed_key, key_block)?;
        let value_panels =
            self.prepare_attention_operand(MidValueId(2), &packed_value, key_block)?;
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
            let k = self.copy(key_panels, key_block_type, vec![0, start, 0]);
            // Flash broadcasts K/V together. Full materialization keeps their
            // large resident matrices in disjoint lifetimes.
            let v = (!materialized)
                .then(|| self.copy(value_panels, value_block_type.clone(), vec![0, start, 0]));
            let scores = self.compute(
                vec![query_buffer, k],
                scores_type.clone(),
                query_key.clone(),
                Some(qk_axes),
                None,
                vec![],
            );
            weights = Some(self.compute(
                vec![scores],
                weights_type.clone(),
                TileKernelSpec::AttentionSoftmax {
                    head_dimension: query.shape.0[2],
                    key_columns: valid,
                    padded_key_columns: key_block,
                },
                None,
                weights,
                vec![],
            ));
            let weights_id = weights?;
            let v =
                v.unwrap_or_else(|| self.copy(value_panels, value_block_type, vec![0, start, 0]));
            let product = self.compute(
                vec![weights_id, v],
                product_type.clone(),
                probability_value.clone(),
                Some(pv_axes),
                None,
                vec![
                    OperandWindow(vec![(2, 0, key_block)]),
                    OperandWindow::default(),
                ],
            );
            result = Some(self.compute(
                vec![product, weights_id],
                output.clone(),
                TileKernelSpec::AttentionMerge {
                    value_dimension: output.shape.0[2],
                    padded_value_dimension: value_width,
                    key_block_columns: key_block,
                    initial: start == 0,
                    final_block: materialized || start + key_block >= key_rows,
                },
                None,
                result,
                vec![],
            ));
        }
        result
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
        let preparation_rows =
            if resident.format.layout.order == ElementOrder::Amp(AmpOrder::TransposedRight) {
                key_block.min(AMP_INNER_BLOCK)
            } else {
                key_block
            };
        let heads = u16::try_from(resident.shape.0[0]).ok()?;
        let blocks = u16::try_from(resident.shape.0[1].div_ceil(preparation_rows))
            .ok()?
            .min(resident.format.layout.tiling.tile_count / heads)
            .max(1);
        let columns = u16::try_from(resident.shape.0[2].div_ceil(AMP_COLUMN_MICRO))
            .ok()?
            .min(resident.format.layout.tiling.tile_count / heads / blocks)
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
