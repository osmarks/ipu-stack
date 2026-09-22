use super::super::candidates::LiveValues;
use super::super::{SearchLimits, boundary_layouts, plan};
use super::*;
use crate::planner::tests::gelu;
use ipu_target::Target;

#[test]
fn large_gemm_catalogues_offer_head_splits_orders_and_reduce_scatter() {
    for (groups, m, k, n, precision) in [
        (1, 729, 1152, 4304, Precision::F8F143 { scale_exponent: -2 }),
        (16, 729, 72, 729, Precision::F16),
    ] {
        let input = |shape| TensorType::new(shape, precision, Layout::row_sharded(1));
        let a = input(vec![groups, m, k]);
        let b = input(vec![groups, k, n]);
        let start = std::time::Instant::now();
        let choices = choices(
            [&a, &b],
            &TensorShape(vec![groups, m, n]),
            GemmOptions::default(),
            1472,
        )
        .unwrap();
        eprintln!(
            "G={groups} M={m} K={k} N={n}: {} choices in {:?}",
            choices.len(),
            start.elapsed()
        );
        assert!(choices.iter().any(|c| c.swapped));
        assert!(choices.iter().any(|c| !c.swapped));
        assert!(choices.iter().any(|c| c.operands.iter().any(|t| {
            t.format
                .layout
                .tiling
                .axes
                .iter()
                .any(|a| !a.partitions.is_power_of_two())
        })));
        assert!(
            choices
                .iter()
                .any(|c| c.result.format.layout.tiling.axes[0].tile_stride
                    < c.result.format.layout.tiling.axes[1].tile_stride)
        );
        assert!(
            choices
                .iter()
                .any(|c| c.result.format.layout.tiling.axes[0].tile_stride
                    > c.result.format.layout.tiling.axes[1].tile_stride)
        );
        assert!(choices.iter().any(|c| {
            c.reduction
                .as_ref()
                .is_some_and(|r| r.tiling.tile_count == c.result.format.layout.tiling.tile_count)
        }));
        if groups > 1 {
            assert!(choices.iter().any(|c| {
                c.result
                    .format
                    .layout
                    .tiling
                    .axes
                    .iter()
                    .any(|a| a.axis == TensorAxis::FromEnd(3) && u32::from(a.partitions) == groups)
            }));
        }
        // Check every actual owner map, including broadcast input replicas.
        for c in &choices {
            for t in c.operands.iter().chain(std::iter::once(&c.result)) {
                t.format.layout.resolve(&t.shape).unwrap();
            }
        }
    }
}

#[test]
fn gemm_construction_accepts_private_mid_results_without_high_nodes() {
    let mut rng = fastrand::Rng::with_seed(0x1a7e_24ed);
    for _ in 0..24 {
        let (m, k) = (rng.u32(1..=8), rng.u32(1..=2) * 16);
        let mut high = HighGraph::new();
        let a = high.host_input("a", [m, k]).unwrap();
        let b = high.parameter("b", [k, k]).unwrap();
        let output = high.gemm(a, b).unwrap();
        let source = high.operations()[0].id;
        let options = GemmOptions::default();
        let shape = TensorShape(vec![m, k]);
        let live = [a, b]
            .into_iter()
            .zip([[m, k], [k, k]])
            .map(|(id, shape)| {
                (
                    id,
                    super::super::candidates::BoundaryValue {
                        tensor: TensorType::new(shape, Precision::F16, Layout::row_sharded(1)),
                        owners: OwnerMap::default(),
                    },
                )
            })
            .collect();
        let mut prefix = Candidate::inputs(&high, &live, 4, 1);
        let inputs = [prefix.bindings[&a], prefix.bindings[&b]];
        let first = choices([&live[&a].tensor, &live[&b].tensor], &shape, options, 4).unwrap();
        let private = append(
            &mut prefix.graph,
            inputs,
            &first[rng.usize(..first.len())],
            options,
            &shape,
            source,
            output,
            None,
        );
        assert!(!prefix.bindings.values().any(|&id| id == private));
        let second = choices(
            [
                &prefix.graph.values[private.index() as usize].tensor_type,
                &prefix.graph.values[inputs[1].index() as usize].tensor_type,
            ],
            &shape,
            options,
            4,
        )
        .unwrap();
        let data = [m * k, k * k].map(|count| {
            (0..count)
                .map(|_| f64::from(rng.i32(-8..=8)) / 16.0)
                .collect::<Vec<_>>()
        });
        let dense = |input: &[f64]| {
            (0..m * k)
                .map(|i| {
                    (0..k)
                        .map(|j| {
                            input[((i / k) * k + j) as usize] * data[1][(j * k + i % k) as usize]
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        let expected = dense(&dense(&data[0]));
        for choice in second {
            let mut candidate = Candidate {
                graph: prefix.graph.clone(),
                bindings: prefix.bindings.clone(),
                end: 1,
            };
            let result = append(
                &mut candidate.graph,
                [private, inputs[1]],
                &choice,
                options,
                &shape,
                source,
                output,
                Some(&Layout::row_sharded(1)),
            );
            candidate.graph.outputs = vec![result];
            candidate.graph.validate().unwrap();
            let actual = evaluate(&candidate, &data, &[a, b]);
            assert_eq!(actual[result.index() as usize], expected);
            crate::low::expand::expand_tiles(Target::Ipu21, &candidate.graph, false).unwrap();
            assert_eq!(
                &candidate.graph.operations[..prefix.graph.operations.len()],
                prefix.graph.operations
            );
            assert_eq!(
                &candidate.graph.values[..prefix.graph.values.len()],
                prefix.graph.values
            );
        }
    }
}

fn index(shape: &[u32], coordinates: &[u32]) -> usize {
    shape
        .iter()
        .zip(coordinates)
        .fold(0, |i, (&width, &coordinate)| {
            i * width as usize + coordinate as usize
        })
}

fn coordinates(shape: &[u32], mut index: usize) -> Vec<u32> {
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = (index % shape[axis] as usize) as u32;
        index /= shape[axis] as usize;
    }
    result
}

// Interpret the emitted distributed operations, including actual ownership and
// windows. Compare against an ordinary dense product, independent of its grid.
fn evaluate(candidate: &Candidate, data: &[Vec<f64>; 2], origins: &[ValueId]) -> Vec<Vec<f64>> {
    let graph = &candidate.graph;
    let mut values = graph
        .values
        .iter()
        .map(|v| vec![0.0; v.tensor_type.shape.elements() as usize])
        .collect::<Vec<_>>();
    for input in &graph.inputs {
        let value = &graph.values[input.value.index() as usize];
        values[input.value.index() as usize] =
            data[origins.iter().position(|&id| id == value.origin).unwrap()].clone();
    }
    for op in &graph.operations {
        let output = op.results[0].index() as usize;
        let tensor = &graph.values[output].tensor_type;
        let mut result = op.output_aliases.first().map_or_else(
            || vec![0.0; values[output].len()],
            |&(_, input)| values[op.inputs[input].index() as usize].clone(),
        );
        match &op.kind {
            MidOperationKind::Cast { .. } => {
                result.clone_from(&values[op.inputs[0].index() as usize]);
            }
            MidOperationKind::Gelu => {
                for (out, &x) in result
                    .iter_mut()
                    .zip(&values[op.inputs[0].index() as usize])
                {
                    *out = gelu(x);
                }
            }
            MidOperationKind::Copy { mapping, .. } => {
                assert!(mapping.view.is_none());
                let input = op.inputs[0].index() as usize;
                for (i, out) in result.iter_mut().enumerate() {
                    let mut c = coordinates(&tensor.shape.0, i);
                    for (c, offset) in c.iter_mut().zip(&mapping.offsets) {
                        *c += offset;
                    }
                    *out = values[input][index(&graph.values[input].tensor_type.shape.0, &c)];
                }
            }
            MidOperationKind::Gemm { axes, mode, .. } => {
                let a = op.inputs[0].index() as usize;
                let b = op.inputs[1].index() as usize;
                let left = &graph.values[a].tensor_type;
                let right = &graph.values[b].tensor_type;
                let li = axes.left_inner.resolve(left.shape.0.len()).unwrap();
                let ri = axes.right_inner.resolve(right.shape.0.len()).unwrap();
                let lr = 2 * left.shape.0.len() - 3 - li;
                let rc = 2 * right.shape.0.len() - 3 - ri;
                let left_shards = left.format.layout.shard_extents(&left.shape).unwrap();
                let right_shards = right.format.layout.shard_extents(&right.shape).unwrap();
                for (tile, extents) in tensor.format.layout.shard_extents(&tensor.shape).unwrap() {
                    let out = &extents;
                    let select = |operand: usize, shards: &[(u16, Vec<crate::ShardExtent>)]| {
                        let extents = &shards.iter().find(|(owner, _)| *owner == tile).unwrap().1;
                        let OperandIndexing::Local(window) = &op.operands[operand] else {
                            panic!()
                        };
                        window.select(extents, false).unwrap()
                    };
                    let le = select(0, &left_shards);
                    let re = select(1, &right_shards);
                    for (i, result) in result.iter_mut().enumerate() {
                        let c = coordinates(&tensor.shape.0, i);
                        if !c
                            .iter()
                            .zip(out)
                            .all(|(&c, e)| c >= e.start && c < e.logical_end)
                        {
                            continue;
                        }
                        let mut lc = le.iter().map(|e| e.start).collect::<Vec<_>>();
                        let mut rc0 = re.iter().map(|e| e.start).collect::<Vec<_>>();
                        for (coords, shape) in
                            [(&mut lc, &left.shape.0), (&mut rc0, &right.shape.0)]
                        {
                            for axis in 0..shape.len() - 2 {
                                coords[axis] = if shape[axis] == 1 {
                                    0
                                } else {
                                    c[c.len() - shape.len() + axis]
                                };
                            }
                        }
                        let oc = axes.output_column.resolve(c.len()).unwrap();
                        let or = 2 * c.len() - 3 - oc;
                        lc[lr] = c[or];
                        rc0[rc] = c[oc];
                        let mut sum = 0.0;
                        for k in le[li].start.max(re[ri].start)
                            ..le[li].logical_end.min(re[ri].logical_end)
                        {
                            lc[li] = k;
                            rc0[ri] = k;
                            sum += values[a][index(&left.shape.0, &lc)]
                                * values[b][index(&right.shape.0, &rc0)];
                        }
                        if *mode == GemmKernelMode::Accumulate {
                            *result += sum;
                        } else {
                            *result = sum;
                        }
                    }
                }
            }
            MidOperationKind::ReductionSum { partials } => {
                let seed = &values[op.inputs[0].index() as usize];
                let rest = &values[op.inputs[1].index() as usize];
                let count = result.len();
                for (i, out) in result.iter_mut().enumerate() {
                    *out = seed[i]
                        + (0..usize::from(*partials - 1))
                            .map(|p| rest[p * count + i])
                            .sum::<f64>();
                }
            }
            other => panic!("unexpected operation {other:?}"),
        }
        values[output] = result;
    }
    values
}

#[test]
fn neighbour_layouts_connect_gemm_and_elementwise_chains_without_importing_residuals() {
    let mut rng = fastrand::Rng::with_seed(0xca7a_1090);
    for _ in 0..8 {
        let m = rng.u32(2..=6);
        let k = 16 * rng.u32(1..=2);
        let mut high = HighGraph::new();
        let x = high.host_input("x", [m, k]).unwrap();
        let w = high.parameter("w", [k, k]).unwrap();
        let first = high.gelu(x).unwrap();
        let product = high.gemm(first, w).unwrap();
        let middle = high.gelu(product).unwrap();
        let output = high.gemm(middle, w).unwrap();
        high.set_outputs([output, x]).unwrap();
        let mut config = PipelineConfig::new(Target::Ipu21, 2).with_input(
            x,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            },
        );
        config.standard_memory_reservation_bytes = 0;
        let layouts = boundary_layouts(&high, &config);
        let catalogue = super::super::candidates::catalogue(&high, &layouts, &config).unwrap();
        for (position, alternatives) in catalogue.iter().enumerate() {
            for candidate in alternatives {
                let imports = candidate
                    .graph
                    .inputs
                    .iter()
                    .map(|input| candidate.graph.values[input.value.index() as usize].origin)
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(
                    imports,
                    high.operations()[position].inputs.iter().copied().collect()
                );
            }
        }
        // The consumer's required packed layout reaches the preceding GeLU,
        // although neither the input nor the default output uses that layout.
        for consumer in &catalogue[1] {
            let required = &consumer.graph.values[consumer.bindings[&first].index() as usize];
            assert!(catalogue[0].iter().any(|producer| {
                let output = &producer.graph.values[producer.graph.outputs[0].index() as usize];
                output.tensor_type == required.tensor_type
            }));
        }
        let exact = SearchLimits {
            states_per_boundary: None,
            paths_per_state: None,
        };
        let graph = plan(&high, &layouts, &config, exact).unwrap();
        let mut constrained = layouts.clone();
        for id in [first, middle] {
            constrained.insert(id, Some(Layout::row_sharded(1)));
        }
        let conventional = plan(&high, &constrained, &config, exact).unwrap();
        assert!(graph.estimated_cycles <= conventional.estimated_cycles);
        let data = [m * k, k * k].map(|count| {
            (0..count)
                .map(|_| f64::from(rng.i32(-8..=8)) / 16.0)
                .collect::<Vec<_>>()
        });
        let dense = |input: &[f64]| {
            (0..m * k)
                .map(|i| {
                    (0..k)
                        .map(|j| {
                            input[((i / k) * k + j) as usize] * data[1][(j * k + i % k) as usize]
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        };
        let expected = dense(
            &dense(&data[0].iter().copied().map(gelu).collect::<Vec<_>>())
                .into_iter()
                .map(gelu)
                .collect::<Vec<_>>(),
        );
        let candidate = Candidate {
            end: 4,
            graph,
            bindings: Default::default(),
        };
        let actual = evaluate(&candidate, &data, &[x, w]);
        for (&actual, expected) in actual[candidate.graph.outputs[0].index() as usize]
            .iter()
            .zip(expected)
        {
            assert!((actual - expected).abs() < 1e-10);
        }
        assert_eq!(actual[candidate.graph.outputs[1].index() as usize], data[0]);
        crate::low::expand::expand_tiles(Target::Ipu21, &candidate.graph, false).unwrap();
    }
}

#[test]
fn randomized_distributed_gemms_match_dense_products_and_lower() {
    let mut rng = fastrand::Rng::with_seed(0x6e44_2026);
    for case in 0..48 {
        let (m, k, n) = (rng.u32(1..=8), rng.u32(1..=48), rng.u32(1..=48));
        let options = GemmOptions {
            transpose_left: rng.bool(),
            transpose_right: rng.bool(),
        };
        let mut ls = if options.transpose_left {
            vec![k, m]
        } else {
            vec![m, k]
        };
        let mut rs = if options.transpose_right {
            vec![n, k]
        } else {
            vec![k, n]
        };
        if case % 3 == 0 {
            ls.insert(0, 2);
        }
        if case % 4 == 0 {
            rs.insert(0, 2);
        }
        if case % 5 == 0 {
            if ls.len() == 2 {
                ls.insert(0, 1);
            }
            if rs.len() == 2 {
                rs.insert(0, 1);
            }
            ls.insert(0, 1);
            rs.insert(0, 3);
        }
        let mut high = HighGraph::new();
        let a = high.host_input("a", ls.clone()).unwrap();
        let b = high.parameter("b", rs.clone()).unwrap();
        let output = high.gemm_with_options(a, b, options).unwrap();
        high.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(Target::Ipu21, 4);
        config.standard_memory_reservation_bytes = 0;
        let mut live = LiveValues::new();
        for (operand, (&id, shape)) in [a, b].iter().zip([&ls, &rs]).enumerate() {
            let tensor = TensorType::new(
                shape.iter().copied(),
                if case % 2 == 0 || (case % 3 == 0 && operand == 0) {
                    Precision::F16
                } else {
                    Precision::F8F143 { scale_exponent: -2 }
                },
                if case % 2 == 0 {
                    Layout::logical_linear(1, 1)
                } else {
                    operand_layout(
                        operand == 0,
                        if operand == 0 {
                            options.transpose_left
                        } else {
                            options.transpose_right
                        },
                        1,
                        1,
                        1,
                        1,
                        32,
                        16,
                    )
                },
            );
            config.inputs.insert(id, tensor.format.clone());
            live.insert(
                id,
                super::super::candidates::BoundaryValue {
                    tensor,
                    owners: OwnerMap::default(),
                },
            );
        }
        if case % 4 == 0 {
            config.inputs.remove(&b);
            live.get_mut(&b).unwrap().tensor.format =
                super::super::parameters::default_format(&high, b, &config).unwrap();
        }
        let mut choices = boundary_layouts(&high, &config);
        let selectable = if case % 3 == 0 || case % 4 == 0 {
            choices.insert(b, None);
            vec![b]
        } else {
            Vec::new()
        };
        if case % 6 == 0 {
            choices.insert(output, Some(Layout::row_sharded(1)));
        }
        let data = [ls.iter().product::<u32>(), rs.iter().product::<u32>()].map(|count| {
            (0..count)
                .map(|_| f64::from(rng.i32(-16..=16)) / 16.0)
                .collect::<Vec<_>>()
        });
        let shape = high.value_shape(output).unwrap();
        let reference = (0..shape.elements() as usize)
            .map(|i| {
                let c = coordinates(&shape.0, i);
                let row = c[c.len() - 2];
                let column = c[c.len() - 1];
                (0..k)
                    .map(|inner| {
                        let mut lc = if options.transpose_left {
                            vec![inner, row]
                        } else {
                            vec![row, inner]
                        };
                        let mut rc = if options.transpose_right {
                            vec![column, inner]
                        } else {
                            vec![inner, column]
                        };
                        for (coords, input_shape) in [(&mut lc, &ls), (&mut rc, &rs)] {
                            for axis in (0..input_shape.len() - 2).rev() {
                                coords.insert(
                                    0,
                                    if input_shape[axis] == 1 {
                                        0
                                    } else {
                                        c[c.len() - input_shape.len() + axis]
                                    },
                                );
                            }
                        }
                        data[0][index(&ls, &lc)] * data[1][index(&rs, &rc)]
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let catalogue = super::super::candidates::catalogue(&high, &choices, &config).unwrap();
        let mut candidates = catalogue[0]
            .iter()
            .flat_map(|candidate| {
                super::super::search::connect(&high, 0, &live, candidate, &selectable, &config)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(!candidates.is_empty());
        let mut best = u64::MAX;
        for candidate in &mut candidates {
            candidate.graph.outputs = vec![candidate.bindings[&output]];
            let values = evaluate(candidate, &data, &[a, b]);
            assert_eq!(
                values[candidate.bindings[&output].index() as usize],
                reference,
                "case={case}"
            );
            let low = crate::low::expand::expand_tiles(Target::Ipu21, &candidate.graph, false)
                .unwrap_or_else(|e| {
                    for end in 1..=candidate.graph.operations.len() {
                        let mut prefix = candidate.graph.clone();
                        prefix.operations.truncate(end);
                        prefix.outputs = prefix.operations[end - 1].results.clone();
                        if let Err(err) =
                            crate::low::expand::expand_tiles(Target::Ipu21, &prefix, false)
                        {
                            panic!(
                                "case={case}: first failure at {end}: {err:?} {:?}\n{:#?}",
                                prefix.operations[end - 1],
                                prefix.values
                            );
                        }
                    }
                    panic!("case={case}: {e:?}")
                });
            assert_eq!(
                low.kernel_runs
                    .iter()
                    .filter_map(|run| run.product_flops)
                    .map(|flops| flops[0])
                    .sum::<u64>(),
                2 * shape.elements() * u64::from(k),
                "every logical product must be computed exactly once: case={case}"
            );
            candidate.graph.refresh_estimates(Target::Ipu21).unwrap();
            best = best.min(candidate.graph.estimated_cycles);
        }
        let selected = plan(
            &high,
            &choices,
            &config,
            SearchLimits {
                states_per_boundary: None,
                paths_per_state: None,
            },
        )
        .unwrap();
        assert_eq!(selected.estimated_cycles, best);
        for budget in [512, 2048, 8192] {
            config.tile_memory_budget_bytes = budget;
            match plan(&high, &choices, &config, SearchLimits::default()) {
                Ok(graph) => {
                    assert!(graph.peak_memory.total <= budget);
                    crate::low::expand::expand_tiles(Target::Ipu21, &graph, false).unwrap();
                }
                Err(PlanningError::NoPlan(_)) => {}
                Err(e) => panic!("case={case}: {e}"),
            }
        }
    }
}
