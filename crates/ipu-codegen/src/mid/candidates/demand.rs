//! Backward layout requests. These describe consumer-compatible storage families,
//! not selected implementations or execution costs. Views and layout-preserving
//! GELU pass requests upstream; the forward planner prices every alternative.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::mid) struct OutputDemand {
    pub order: ElementOrder,
    pub column_groups: u16,
    pub inner_grain: u32,
}

impl OutputDemand {
    pub(in crate::mid) fn matches(self, layout: &Layout, shape: &TensorShape) -> bool {
        let order_matches = layout.order == self.order
            || layout.order.micro_panel_order().is_some()
                && layout.order.micro_panel_order() == self.order.micro_panel_order();
        let grain_matches = self.inner_grain == 1
            || layout.resolve(shape).ok().is_some_and(|resolved| {
                resolved.axes().is_some_and(|axes| {
                    let from_end = if self.order.micro_panel_order()
                        == Some(MicroPanelOrder::ColumnsThenRows)
                    {
                        2
                    } else {
                        1
                    };
                    axes.len().checked_sub(from_end).is_some_and(|axis| {
                        axes[axis].complete_panels_except_tail(self.inner_grain)
                    }) && axes
                        .len()
                        .checked_sub(from_end)
                        .is_some_and(|axis| axes[axis].maximum_extent() >= self.inner_grain)
                })
            });
        order_matches
            && grain_matches
            && (self.column_groups == 1
                || layout.tiling.axes.iter().any(|axis| {
                    axis.axis == TensorAxis::FromEnd(1)
                        && (axis.padding_groups.is_multiple_of(self.column_groups)
                            || shape.0.last().is_some_and(|&width| {
                                axis.partitions.is_multiple_of(self.column_groups)
                                    && width.is_multiple_of(
                                        u32::from(self.column_groups) * axis.block_size,
                                    )
                                    && (width / axis.block_size)
                                        .is_multiple_of(u32::from(axis.partitions))
                            }))
                }))
    }

    fn through_view(self, view: AxisFactorView, shape: &TensorShape) -> Option<Self> {
        if view.factor == 1 {
            return Some(self);
        }
        // Factoring a matrix column into a batch axis leaves the within-panel
        // orientation unchanged, but introduces independently padded groups.
        // Other axis mappings need a richer output ABI; do not guess an order.
        if view.reversed
            || view.split_axis != shape.0.len().checked_sub(1)?
            || view.merge_axis >= shape.0.len().checked_sub(2)?
        {
            return None;
        }
        view.output_shape(shape)?;
        Some(Self {
            column_groups: self
                .column_groups
                .checked_mul(u16::try_from(view.factor).ok()?)?,
            ..self
        })
    }
}

#[derive(Default)]
pub(in crate::mid) struct OutputDemands(BTreeMap<ValueId, Vec<OutputDemand>>);

impl OutputDemands {
    pub(in crate::mid) fn new(
        operations: &[Operation],
        shapes: &BTreeMap<ValueId, TensorShape>,
        config: &PipelineConfig,
    ) -> Self {
        let mut requests = Self::default();
        for operation in operations.iter().rev() {
            if matches!(operation.kind, OperationKind::Gelu) {
                if let ([input], [output]) =
                    (operation.inputs.as_slice(), operation.results.as_slice())
                {
                    for demand in requests.get(*output).to_vec() {
                        requests.insert(*input, demand);
                    }
                }
                continue;
            }
            if matches!(operation.kind, OperationKind::Gemm(_))
                && config.operator_candidates.iter().any(|candidate| {
                    matches!(
                        candidate.operator(),
                        MidOperator::Gemm {
                            multiply: Precision::F8F143 { .. },
                            ..
                        }
                    )
                })
            {
                for (index, &input) in operation.inputs.iter().enumerate() {
                    requests.insert(
                        input,
                        OutputDemand {
                            order: ElementOrder::Amp(if index == 0 {
                                AmpOrder::Left
                            } else {
                                AmpOrder::TransposedLeft
                            }),
                            column_groups: 1,
                            inner_grain: 32,
                        },
                    );
                }
            }
            if let OperationKind::View(view) = operation.kind {
                if let ([input], [output]) =
                    (operation.inputs.as_slice(), operation.results.as_slice())
                    && let Some(shape) = shapes.get(input)
                {
                    for demand in requests.get(*output).to_vec() {
                        if let Some(demand) = demand.through_view(view, shape) {
                            requests.insert(*input, demand);
                        }
                    }
                }
                continue;
            }
            for (index, &input) in operation.inputs.iter().enumerate() {
                let Some(shape) = shapes.get(&input) else {
                    continue;
                };
                let layouts =
                    direct_consumer_layouts(std::slice::from_ref(operation), input, shape, config);
                for layout in layouts {
                    requests.insert(
                        input,
                        OutputDemand {
                            order: layout.order,
                            column_groups: 1,
                            inner_grain: 1,
                        },
                    );
                }
                // Static operator families expose their requirements directly.
                // Preserve-input policies have no concrete backward requirement.
                for candidate in &config.operator_candidates {
                    if operator_matches(&operation.kind, candidate.operator())
                        && candidate.format_policy() == OperatorFormatPolicy::Concrete
                        && let Some(candidate) = candidate.concrete()
                        && let Some(requirement) = candidate.plan.requirements.inputs.get(index)
                    {
                        requests.insert(
                            input,
                            OutputDemand {
                                order: requirement.format.layout.order,
                                column_groups: 1,
                                inner_grain: 1,
                            },
                        );
                    }
                }
            }
        }
        requests
    }

    fn insert(&mut self, value: ValueId, demand: OutputDemand) {
        let requests = self.0.entry(value).or_default();
        if !requests.contains(&demand) {
            requests.push(demand);
            requests.sort();
        }
    }

    pub(in crate::mid) fn get(&self, value: ValueId) -> &[OutputDemand] {
        self.0.get(&value).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_cross_view_chains_and_fanout_but_stop_at_compute() {
        let mut graph = ComputeGraph::new();
        let x = graph.host_input("x", [1, 17, 64]).unwrap();
        let w = graph.parameter("w", [64, 288]).unwrap();
        let projection = graph.gemm(x, w).unwrap();
        let two = graph.split_heads(projection, 2).unwrap();
        let four = graph.split_heads(two, 2).unwrap();
        let result = graph.flash_attention(four, four, four).unwrap();
        graph.set_outputs([result]).unwrap();
        let config = PipelineConfig::new(64);
        let demands = OutputDemands::new(graph.operations(), graph.value_shapes(), &config);
        for (value, groups) in [(projection, 4), (two, 2), (four, 1)] {
            let requests = demands.get(value);
            assert!(requests.iter().any(|d| d.order.micro_panel_order()
                == Some(MicroPanelOrder::RowsThenColumns)
                && d.column_groups == groups));
            assert!(requests.iter().any(|d| d.order.micro_panel_order()
                == Some(MicroPanelOrder::ColumnsThenRows)
                && d.column_groups == groups));
            assert_eq!(
                requests.iter().collect::<BTreeSet<_>>().len(),
                requests.len()
            );
        }
        // GEMM may request its own operand layouts, but the attention's head
        // grouping must not leak through arithmetic to the host input.
        assert!(demands.get(x).iter().all(|d| d.column_groups == 1));
    }

    #[test]
    fn unsupported_axis_mappings_do_not_invent_a_packed_order() {
        let demand = OutputDemand {
            order: ElementOrder::Amp(AmpOrder::Left),
            column_groups: 1,
            inner_grain: 1,
        };
        assert!(
            demand
                .through_view(AxisFactorView::new(1, 2, 2), &TensorShape::new([1, 16, 32]))
                .is_none()
        );
        assert!(
            demand
                .through_view(AxisFactorView::new(2, 0, 3), &TensorShape::new([1, 16, 32]))
                .is_none()
        );
    }
}
