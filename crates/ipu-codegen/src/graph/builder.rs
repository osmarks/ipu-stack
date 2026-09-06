//! One inherent operation-building API for top-level graphs and repeat bodies.
//! Both builders provide the same value/shape tables and operation-ID counters.

macro_rules! graph_operations {
    () => {
        pub fn gemm(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
            self.gemm_with_options(left, right, GemmOptions::default())
        }

        pub fn gemm_with_options(
            &mut self,
            left: ValueId,
            right: ValueId,
            options: GemmOptions,
        ) -> GraphResult<ValueId> {
            self.inferred_result(OperationKind::Gemm(options), [left, right])
        }

        pub fn gelu(&mut self, input: ValueId) -> GraphResult<ValueId> {
            self.inferred_result(OperationKind::Gelu, [input])
        }

        pub fn add(&mut self, left: ValueId, right: ValueId) -> GraphResult<ValueId> {
            self.inferred_result(OperationKind::Add, [left, right])
        }

        /// Apply a logical axis split/merge; materialization is selected by planning.
        pub fn view(&mut self, input: ValueId, view: AxisFactorView) -> GraphResult<ValueId> {
            self.inferred_result(OperationKind::View(view), [input])
        }

        /// Convert `[batch, rows, heads * channels]` into attention streams.
        pub fn split_heads(&mut self, input: ValueId, heads: u32) -> GraphResult<ValueId> {
            if self
                .shapes
                .get(&input)
                .is_some_and(|shape| shape.0.len() != 3)
            {
                return Err(GraphError::InvalidShape(
                    "split_heads requires [batch, rows, channels] input".into(),
                ));
            }
            self.view(input, AxisFactorView::new(2, 0, heads))
        }

        pub fn flash_attention(
            &mut self,
            query: ValueId,
            key: ValueId,
            value: ValueId,
        ) -> GraphResult<ValueId> {
            self.flash_attention_with_options(query, key, value, AttentionOptions::default())
        }

        pub fn flash_attention_with_options(
            &mut self,
            query: ValueId,
            key: ValueId,
            value: ValueId,
            options: AttentionOptions,
        ) -> GraphResult<ValueId> {
            self.inferred_result(OperationKind::FlashAttention(options), [query, key, value])
        }

        pub fn operation(
            &mut self,
            kind: OperationKind,
            inputs: impl IntoIterator<Item = ValueId>,
            result_shapes: impl IntoIterator<Item = TensorShape>,
        ) -> GraphResult<Vec<ValueId>> {
            append_operation(
                &mut self.operations,
                &mut self.values,
                &mut self.shapes,
                &mut self.next_operation,
                &mut self.next_value,
                kind,
                inputs,
                result_shapes,
            )
        }

        fn inferred_result(
            &mut self,
            kind: OperationKind,
            inputs: impl IntoIterator<Item = ValueId>,
        ) -> GraphResult<ValueId> {
            let inputs = inputs.into_iter().collect::<Vec<_>>();
            validate_inputs(&self.values, &inputs)?;
            let shape = infer_shape(&kind, &inputs, &self.shapes)?;
            Ok(self
                .operation(kind, inputs, [shape])?
                .into_iter()
                .next()
                .expect("one result was requested"))
        }
    };
}

pub(super) use graph_operations;
