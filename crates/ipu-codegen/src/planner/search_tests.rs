use super::tests::gelu;
use super::*;
use crate::mid::MidOperationKind;
use crate::{Precision, TensorFormat};
use ipu_target::Target;

const EXACT: SearchLimits = SearchLimits {
    states_per_boundary: None,
    paths_per_state: None,
};

fn fixture() -> (HighGraph, PipelineConfig) {
    let mut high = HighGraph::new();
    let x = high.host_input("x", [16, 16]).unwrap();
    let bias = high.parameter("bias", [16, 16]).unwrap();
    let add = high.add(x, bias).unwrap();
    let gelu = high.gelu(add).unwrap();
    let residual = high.add(gelu, x).unwrap();
    let output = high.gelu(residual).unwrap();
    high.set_outputs([output]).unwrap();
    let mut config = PipelineConfig::new(Target::Ipu21, 4);
    config.standard_memory_reservation_bytes = 0;
    for input in [x, bias] {
        config.inputs.insert(
            input,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            },
        );
    }
    (high, config)
}

fn execute(graph: &MidGraph) -> Vec<Vec<f64>> {
    let mut values = vec![Vec::new(); graph.values.len()];
    for input in &graph.inputs {
        values[input.value.index() as usize] = (0..256)
            .map(|i| {
                if input.name == "x" {
                    (i as f64 - 100.0) / 32.0
                } else {
                    (i % 7) as f64 / 16.0
                }
            })
            .collect();
    }
    for op in &graph.operations {
        let left = &values[op.inputs[0].index() as usize];
        let output = match op.kind {
            MidOperationKind::Copy { .. } => left.clone(),
            MidOperationKind::Add | MidOperationKind::BiasGelu => left
                .iter()
                .zip(&values[op.inputs[1].index() as usize])
                .map(|(a, b)| {
                    if matches!(op.kind, MidOperationKind::BiasGelu) {
                        gelu(a + b)
                    } else {
                        a + b
                    }
                })
                .collect(),
            MidOperationKind::Gelu => left.iter().copied().map(gelu).collect(),
            _ => panic!("unexpected test kernel"),
        };
        values[op.results[0].index() as usize] = output;
    }
    graph
        .outputs
        .iter()
        .map(|id| values[id.index() as usize].clone())
        .collect()
}

#[test]
fn dp_matches_exhaustive_boundary_choices_and_preserves_residual_computation() {
    let (high, config) = fixture();
    let choices = boundary_layouts(&high, &config);
    let selected = plan(&high, &choices, &config, EXACT).unwrap();
    let mut best = u64::MAX;
    let mut reference = None;
    // Independent full assignments prevent the DP from merging different
    // layout histories. Compare against their exhaustive minimum.
    for bits in 0..1 << high.operations().len() {
        let mut fixed = choices.clone();
        for (index, op) in high.operations().iter().enumerate() {
            fixed.insert(
                op.results[0],
                Some(Layout::row_sharded(if bits & (1 << index) == 0 {
                    1
                } else {
                    4
                })),
            );
        }
        let program = plan(&high, &fixed, &config, EXACT).unwrap();
        best = best.min(program.estimated_cycles);
        let output = execute(&program);
        if let Some(reference) = &reference {
            assert_eq!(&output, reference);
        }
        reference = Some(output);
    }
    // Unconstrained search can additionally fuse across the otherwise fixed
    // add boundary, so it may beat the best fully constrained assignment.
    assert!(selected.estimated_cycles <= best);
    assert_eq!(execute(&selected), reference.unwrap());
    selected.validate().unwrap();
    // Exercise real binding/lowering, not only the abstract scalar interpreter.
    crate::low::expand::expand_tiles(Target::Ipu21, &selected, false).unwrap();
    for op in &selected.operations {
        assert!(op.source.is_some());
    }
}

#[test]
fn unfused_chain_matches_the_exhaustive_layout_minimum() {
    let mut high = HighGraph::new();
    let x = high.host_input("x", [16, 16]).unwrap();
    let a = high.gelu(x).unwrap();
    let b = high.gelu(a).unwrap();
    let c = high.gelu(b).unwrap();
    high.set_outputs([c]).unwrap();
    let config = PipelineConfig::new(Target::Ipu21, 4).with_input(
        x,
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(1),
        },
    );
    let choices = boundary_layouts(&high, &config);
    let selected = plan(&high, &choices, &config, EXACT).unwrap();
    let best = (0..8)
        .map(|bits| {
            let mut fixed = choices.clone();
            for (index, value) in [a, b, c].into_iter().enumerate() {
                fixed.insert(
                    value,
                    Some(Layout::row_sharded(if bits & (1 << index) == 0 {
                        1
                    } else {
                        4
                    })),
                );
            }
            plan(&high, &fixed, &config, EXACT)
                .unwrap()
                .estimated_cycles
        })
        .min()
        .unwrap();
    assert_eq!(selected.estimated_cycles, best);
}

#[test]
fn escaped_add_result_is_preserved_and_resident_parameters_are_not_freed() {
    let (mut high, config) = fixture();
    let add = high.operations()[0].results[0];
    let final_output = high.outputs()[0];
    high.set_outputs([add, final_output]).unwrap();
    let mut choices = boundary_layouts(&high, &config);
    for op in high.operations() {
        choices.insert(op.results[0], Some(Layout::row_sharded(1)));
    }
    let program = plan(&high, &choices, &config, EXACT).unwrap();
    let outputs = execute(&program);
    for (i, value) in outputs[0].iter().enumerate() {
        assert_eq!(*value, (i as f64 - 100.0) / 32.0 + (i % 7) as f64 / 16.0);
    }
    // Four simultaneous 512-byte allocations: x, bias, saved add, and a
    // subsequent result. A lower limit must not pass by forgetting the residual,
    // saved output, or already-consumed resident parameter.
    assert!(program.peak_memory.total >= 4 * 512);
    let mut tight = config.clone();
    tight.tile_memory_budget_bytes = 4 * 512 - 1;
    assert!(matches!(
        plan(&high, &choices, &tight, EXACT),
        Err(PlanningError::NoPlan(_))
    ));
}

#[test]
fn empty_graph_keeps_identity_outputs_and_checks_memory() {
    let mut high = HighGraph::new();
    let x = high.host_input("x", [16, 16]).unwrap();
    high.host_input("unused", [16, 16]).unwrap();
    high.set_outputs([x, x]).unwrap();
    let mut config = PipelineConfig::new(Target::Ipu21, 1).with_input(
        x,
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(1),
        },
    );
    config.standard_memory_reservation_bytes = 0;
    let choices = boundary_layouts(&high, &config);
    let program = plan(&high, &choices, &config, EXACT).unwrap();
    assert_eq!(program.outputs, vec![program.inputs[0].value; 2]);
    config.tile_memory_budget_bytes = 1;
    assert!(matches!(
        plan(&high, &choices, &config, EXACT),
        Err(PlanningError::NoPlan(0))
    ));
}

#[test]
fn invalid_proposed_layout_does_not_discard_valid_vector_plan() {
    let mut high = HighGraph::new();
    let x = high.host_input("vector", [16]).unwrap();
    let y = high.gelu(x).unwrap();
    high.set_outputs([y]).unwrap();
    let config = PipelineConfig::new(Target::Ipu21, 4).with_input(
        x,
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::logical_linear(1, 1),
        },
    );
    let graph = plan(&high, &boundary_layouts(&high, &config), &config, EXACT).unwrap();
    crate::low::expand::expand_tiles(Target::Ipu21, &graph, false).unwrap();
}

#[test]
fn native_resident_layout_avoids_compact_preparation() {
    let mut high = HighGraph::new();
    let x = high.host_input("x", [16, 16]).unwrap();
    let weight = high.parameter("weight", [16, 16]).unwrap();
    let first = high.gelu(x).unwrap();
    let output = high.add(first, weight).unwrap();
    high.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(Target::Ipu21, 4).with_input(
        x,
        TensorFormat {
            precision: Precision::F16,
            layout: Layout::row_sharded(4),
        },
    );
    let mut choices = boundary_layouts(&high, &config);
    choices.insert(first, Some(Layout::row_sharded(4)));
    choices.insert(output, Some(Layout::row_sharded(4)));
    let selected = plan(&high, &choices, &config, EXACT).unwrap();
    let baseline = parameters::compact_format(high.value_shape(weight).unwrap(), &config);
    choices.insert(weight, Some(baseline.layout));
    let fixed = plan(&high, &choices, &config, EXACT).unwrap();
    assert!(selected.estimated_cycles < fixed.estimated_cycles);
    assert_eq!(execute(&selected), execute(&fixed));
    crate::low::expand::expand_tiles(Target::Ipu21, &selected, false).unwrap();
}

#[test]
fn copies_into_padded_linear_layouts_lower_for_vectors_and_matrices() {
    for shape in [vec![30], vec![3, 10]] {
        let mut high = HighGraph::new();
        let input = high.host_input("input", shape).unwrap();
        let output = high.gelu(input).unwrap();
        high.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(Target::Ipu21, 2).with_input(
            input,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::logical_linear(1, 1),
            },
        );
        let mut layout = Layout::logical_linear(2, 4);
        layout.tiling.axes[0].padding = crate::Padding::Zero;
        let mut choices = boundary_layouts(&high, &config);
        choices.insert(output, Some(layout));
        let mid = plan(&high, &choices, &config, EXACT).unwrap();
        crate::low::expand::expand_tiles(Target::Ipu21, &mid, false).unwrap();
    }
}
