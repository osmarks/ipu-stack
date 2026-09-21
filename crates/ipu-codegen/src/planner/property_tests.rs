//! The oracle enumerates complete paths without Search, dominance, beam limits,
//! or the production fragment appender. It shares candidate generation and the
//! cost model: these tests check selection against that model, not its accuracy.
use super::super::candidates::{self, BoundaryValue, Candidate, LiveValues};
use super::super::{boundary_layouts, plan};
use super::*;
use crate::graph::{GraphInputKind, OperationKind, ValueId};
use crate::mid::{MidInput, MidOperation, MidOperationKind, MidValue, MidValueId, OperandIndexing};
use crate::tensor::{Layout, MemoryClass, OwnerMap, Precision, TensorFormat, TensorType};
use ipu_target::Target;
use std::collections::BTreeSet;

const EXACT: SearchLimits = SearchLimits {
    states_per_boundary: None,
    paths_per_state: None,
};
// Test algorithms with different scratch/cycle costs but the same semantics.
type ScratchChoice = (u32, usize, MemoryClass);

fn initial(high: &HighGraph, config: &PipelineConfig) -> (MidGraph, BTreeMap<ValueId, MidValueId>) {
    let used = high
        .operations()
        .iter()
        .flat_map(|op| high.operation_inputs(op))
        .chain(high.outputs().iter().copied())
        .collect::<BTreeSet<_>>();
    let mut graph = MidGraph {
        tile_count: config.tile_count,
        ..MidGraph::default()
    };
    let mut bindings = BTreeMap::new();
    for input in high
        .inputs()
        .iter()
        .filter(|input| input.kind == GraphInputKind::Parameter || used.contains(&input.value))
    {
        let id = MidValueId::from_index(graph.values.len() as u32);
        graph.values.push(MidValue {
            id,
            origin: input.value,
            storage_group: id,
            owners: OwnerMap::default(),
            tensor_type: TensorType {
                shape: input.shape.clone(),
                format: config.inputs.get(&input.value).cloned().unwrap_or_else(|| {
                    assert_eq!(input.kind, GraphInputKind::Parameter);
                    super::super::parameters::default_format(high, input.value, config).unwrap()
                }),
            },
        });
        graph.inputs.push(MidInput {
            name: input.name.clone(),
            kind: input.kind,
            value: id,
        });
        bindings.insert(input.value, id);
    }
    (graph, bindings)
}

fn live_at(
    high: &HighGraph,
    position: usize,
    graph: &MidGraph,
    bindings: &BTreeMap<ValueId, MidValueId>,
) -> LiveValues {
    // Direct suffix use scan, rather than the DP's birth/last-use intervals.
    high.operations()[position..]
        .iter()
        .flat_map(|op| high.operation_inputs(op))
        .chain(high.outputs().iter().copied())
        .chain(
            high.inputs()
                .iter()
                .filter(|input| input.kind == GraphInputKind::Parameter)
                .filter(|input| {
                    !high
                        .operations()
                        .iter()
                        .any(|op| high.operation_inputs(op).any(|id| id == input.value))
                })
                .map(|input| input.value),
        )
        .filter_map(|high| {
            bindings.get(&high).map(|id| {
                let value = &graph.values[id.index() as usize];
                (
                    high,
                    BoundaryValue {
                        tensor: value.tensor_type.clone(),
                        owners: value.owners.clone(),
                    },
                )
            })
        })
        .collect()
}

fn edges(
    high: &HighGraph,
    position: usize,
    live: &LiveValues,
    choices: &BoundaryLayouts,
    config: &PipelineConfig,
    scratch: &[ScratchChoice],
) -> Vec<Candidate> {
    // Independently derive first-use eligibility for the exhaustive oracle.
    let selectable = high
        .inputs()
        .iter()
        .filter(|input| {
            input.kind == GraphInputKind::Parameter
                && choices.get(&input.value).and_then(Option::as_ref).is_none()
                && high
                    .operation_inputs(&high.operations()[position])
                    .any(|id| id == input.value)
                && !high.operations()[..position]
                    .iter()
                    .any(|op| high.operation_inputs(op).any(|id| id == input.value))
        })
        .map(|input| input.value)
        .collect::<Vec<_>>();
    let candidates =
        candidates::generate(high, position, live, choices, config, &selectable).unwrap();
    if scratch.is_empty() {
        return candidates;
    }
    candidates
        .into_iter()
        .flat_map(|base| {
            scratch.iter().map(move |&(bytes, extra_gelus, class)| {
                let mut candidate = Candidate {
                    end: base.end,
                    graph: base.graph.clone(),
                    bindings: base.bindings.clone(),
                };
                let input = candidate.bindings[&high.operations()[position].inputs[0]];
                let mut setup = Vec::new();
                for delay in 0..=extra_gelus {
                    let id = MidValueId::from_index(candidate.graph.values.len() as u32);
                    let mut value = candidate.graph.values[input.index() as usize].clone();
                    value.id = id;
                    value.storage_group = id;
                    let (kind, inputs, operands) = if delay == 0 {
                        value.tensor_type =
                            TensorType::new([1, bytes / 2], Precision::F16, Layout::row_sharded(1));
                        value.tensor_type.format.layout.memory_class = class;
                        value.owners =
                            OwnerMap::rotated((bytes / 8 % u32::from(config.tile_count)) as u16);
                        (
                            MidOperationKind::FillZero {
                                offset: 0,
                                bytes: 8,
                                padding_only: false,
                            },
                            Vec::new(),
                            Vec::new(),
                        )
                    } else {
                        (
                            MidOperationKind::Gelu,
                            vec![input],
                            vec![OperandIndexing::Elementwise { result: 0 }],
                        )
                    };
                    candidate.graph.values.push(value);
                    setup.push(MidOperation {
                        source: None,
                        inputs,
                        results: vec![id],
                        kind,
                        operands,
                        output_aliases: Vec::new(),
                        output_windows: Vec::new(),
                    });
                }
                setup.append(&mut candidate.graph.operations);
                candidate.graph.operations = setup;
                candidate
            })
        })
        .collect()
}

fn enumerate(
    high: &HighGraph,
    choices: &BoundaryLayouts,
    config: &PipelineConfig,
    scratch: &[ScratchChoice],
) -> Vec<MidGraph> {
    fn visit(
        high: &HighGraph,
        choices: &BoundaryLayouts,
        config: &PipelineConfig,
        scratch: &[ScratchChoice],
        position: usize,
        graph: MidGraph,
        bindings: BTreeMap<ValueId, MidValueId>,
        complete: &mut Vec<MidGraph>,
    ) {
        if position == high.operations().len() {
            let mut graph = graph;
            graph.outputs = high.outputs().iter().map(|id| bindings[id]).collect();
            graph.refresh_estimates(config.target).unwrap();
            complete.push(graph);
            assert!(
                complete.len() < 20_000,
                "test fixture enumerated too many paths"
            );
            return;
        }
        let live = live_at(high, position, &graph, &bindings);
        for candidate in edges(high, position, &live, choices, config, scratch) {
            let mut next = graph.clone();
            let mut bound = bindings.clone();
            let mut ids = BTreeMap::new();
            for input in &candidate.graph.inputs {
                let value = &candidate.graph.values[input.value.index() as usize];
                if input.kind == GraphInputKind::Parameter {
                    let original = &mut next.values[bindings[&value.origin].index() as usize];
                    original.tensor_type = value.tensor_type.clone();
                    original.owners = value.owners.clone();
                }
                ids.insert(
                    input.value,
                    bindings[&candidate.graph.values[input.value.index() as usize].origin],
                );
            }
            // An independent flat-graph concatenation. No append_fragment and
            // no dependency on the DP's exported bindings or reconstruction.
            for value in &candidate.graph.values {
                if !ids.contains_key(&value.id) {
                    let id = MidValueId::from_index(next.values.len() as u32);
                    ids.insert(value.id, id);
                    let mut added = value.clone();
                    added.id = id;
                    added.storage_group = id;
                    next.values.push(added);
                }
            }
            for op in &candidate.graph.operations {
                assert!(op.output_aliases.is_empty());
                assert!(!matches!(op.kind, MidOperationKind::Repeat(_)));
                let mut op = op.clone();
                for id in op.inputs.iter_mut().chain(&mut op.results) {
                    *id = ids[id];
                }
                next.operations.push(op);
            }
            for (&high, &local) in &candidate.bindings {
                bound.insert(high, ids[&local]);
            }
            visit(
                high,
                choices,
                config,
                scratch,
                candidate.end,
                next,
                bound,
                complete,
            );
        }
    }
    let (graph, bindings) = initial(high, config);
    let mut complete = Vec::new();
    visit(
        high,
        choices,
        config,
        scratch,
        0,
        graph,
        bindings,
        &mut complete,
    );
    complete
}

fn explore<'a>(
    high: &'a HighGraph,
    choices: &BoundaryLayouts,
    config: &'a PipelineConfig,
    scratch: &[ScratchChoice],
    limits: SearchLimits,
    reverse: bool,
) -> PlanningResult<Search<'a>> {
    let mut search = Search::new(high, choices, config, limits)?;
    for position in 0..high.operations().len() {
        let mut states = search.take_states(position);
        if reverse {
            states.reverse();
        }
        for state in states {
            let mut alternatives = edges(high, position, &state.live, choices, config, scratch);
            if reverse {
                alternatives.reverse();
            }
            for candidate in alternatives {
                search.extend(position, &state, candidate)?;
            }
        }
    }
    Ok(search)
}

fn optimum(complete: &[MidGraph], config: &PipelineConfig) -> Option<u64> {
    complete
        .iter()
        .filter(|graph| {
            graph.peak_memory.fits_with_budget(
                config.target,
                config.standard_memory_reservation_bytes,
                config.tile_memory_budget_bytes,
            )
        })
        .map(|graph| graph.estimated_cycles)
        .min()
}

fn check_frontier(complete: &[MidGraph], search: &Search<'_>, seed: u64) {
    type Frontiers = HashMap<LiveValues, BTreeSet<[u64; 5]>, foldhash::fast::FixedState>;
    let label = |cycles, peak: MemoryPeaks| {
        [
            cycles,
            peak.standard,
            peak.interleaved,
            peak.total,
            peak.maximum_standard_allocation,
        ]
    };
    let mut expected = Frontiers::default();
    for graph in complete.iter().filter(|g| {
        g.peak_memory.fits_with_budget(
            search.settings.target,
            search.settings.standard_memory_reservation_bytes,
            search.settings.tile_memory_budget_bytes,
        )
    }) {
        let live = graph
            .outputs
            .iter()
            .copied()
            .chain(
                graph
                    .inputs
                    .iter()
                    .filter(|i| i.kind == GraphInputKind::Parameter)
                    .filter(|i| {
                        !search.high.operations().iter().any(|op| {
                            search
                                .high
                                .operation_inputs(op)
                                .any(|id| id == graph.values[i.value.index() as usize].origin)
                        })
                    })
                    .map(|i| i.value),
            )
            .map(|id| {
                let value = &graph.values[id.index() as usize];
                (
                    value.origin,
                    BoundaryValue {
                        tensor: value.tensor_type.clone(),
                        owners: value.owners.clone(),
                    },
                )
            })
            .collect();
        expected
            .entry(live)
            .or_default()
            .insert(label(graph.estimated_cycles, graph.peak_memory));
    }
    for labels in expected.values_mut() {
        // Exhaustive pairwise comparison of completed paths, independent of
        // insertion order and the incremental production dominance predicate.
        let all = labels.clone();
        labels.retain(|a| {
            !all.iter()
                .any(|b| a != b && b.iter().zip(a).all(|(b, a)| b <= a))
        });
    }
    let mut actual = search
        .states
        .last()
        .unwrap()
        .iter()
        .filter(|(_, paths)| !paths.is_empty())
        .map(|(live, paths)| {
            (
                live.clone(),
                paths.iter().map(|p| label(p.cycles, p.peak)).collect(),
            )
        })
        .collect::<Frontiers>();
    // Search retains tile-specific tradeoffs needed by later resident choices.
    // Projecting those labels to global maxima can introduce dominance.
    for labels in actual.values_mut() {
        let all = labels.clone();
        labels.retain(|a| {
            !all.iter()
                .any(|b| a != b && b.iter().zip(a).all(|(b, a)| b <= a))
        });
    }
    assert_eq!(
        actual, expected,
        "frontier mismatch: seed={seed}, budget={}",
        search.settings.tile_memory_budget_bytes
    );
}

fn result_cycles(result: &PlanningResult<MidGraph>) -> Option<u64> {
    match result {
        Ok(graph) => Some(graph.estimated_cycles),
        Err(PlanningError::NoPlan(_)) => None,
        Err(other) => panic!("unexpected planning error: {other:?}"),
    }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

fn check_semantics(high: &HighGraph, mid: &MidGraph, rng: &mut fastrand::Rng) {
    let mut expected = BTreeMap::new();
    for input in high.inputs() {
        expected.insert(
            input.value,
            (0..input.shape.elements())
                .map(|_| rng.f64() * 4.0 - 2.0)
                .collect::<Vec<_>>(),
        );
    }
    let mut actual = vec![Vec::new(); mid.values.len()];
    for input in &mid.inputs {
        actual[input.value.index() as usize] =
            expected[&mid.values[input.value.index() as usize].origin].clone();
    }
    for op in high.operations() {
        let left = &expected[&op.inputs[0]];
        let result = match op.kind {
            OperationKind::Gelu => left.iter().copied().map(gelu).collect(),
            OperationKind::Add => left
                .iter()
                .zip(&expected[&op.inputs[1]])
                .map(|(a, b)| a + b)
                .collect(),
            _ => panic!("unsupported reference operation"),
        };
        expected.insert(op.results[0], result);
    }
    for op in &mid.operations {
        let result = match op.kind {
            MidOperationKind::Copy { .. } => actual[op.inputs[0].index() as usize].clone(),
            MidOperationKind::Gelu => actual[op.inputs[0].index() as usize]
                .iter()
                .copied()
                .map(gelu)
                .collect(),
            MidOperationKind::Add | MidOperationKind::BiasGelu => actual
                [op.inputs[0].index() as usize]
                .iter()
                .zip(&actual[op.inputs[1].index() as usize])
                .map(|(a, b)| {
                    if matches!(op.kind, MidOperationKind::BiasGelu) {
                        gelu(a + b)
                    } else {
                        a + b
                    }
                })
                .collect(),
            MidOperationKind::FillZero { .. } => vec![
                0.0;
                mid.values[op.results[0].index() as usize]
                    .tensor_type
                    .shape
                    .elements() as usize
            ],
            _ => panic!("unsupported mid reference operation"),
        };
        actual[op.results[0].index() as usize] = result;
    }
    assert_eq!(high.outputs().len(), mid.outputs.len());
    for (&high, &mid_id) in high.outputs().iter().zip(&mid.outputs) {
        assert_eq!(mid.values[mid_id.index() as usize].origin, high);
        assert_eq!(actual[mid_id.index() as usize], expected[&high]);
    }
}

fn random_fixture(rng: &mut fastrand::Rng) -> (HighGraph, PipelineConfig, BoundaryLayouts) {
    let tiles = rng.u16(1..=4);
    let shape = if rng.bool() {
        vec![rng.u32(1..=8), 8 * rng.u32(1..=3)]
    } else {
        vec![8 * rng.u32(1..=8)]
    };
    let mut high = HighGraph::new();
    let x = high.host_input("x", shape.clone()).unwrap();
    let weight = high.parameter("w", shape.clone()).unwrap();
    let other = high.host_input("other", shape.clone()).unwrap();
    let mut available = vec![x, weight, other];
    for _ in 0..rng.usize(2..=5) {
        let a = available[rng.usize(..available.len())];
        let value = if rng.bool() {
            high.gelu(a).unwrap()
        } else {
            high.add(a, available[rng.usize(..available.len())])
                .unwrap()
        };
        available.push(value);
    }
    high.set_outputs([
        *available.last().unwrap(),
        available[rng.usize(..available.len())],
    ])
    .unwrap();
    let mut config = PipelineConfig::new(Target::Ipu21, tiles);
    config.standard_memory_reservation_bytes = 0;
    for input in [x, weight, other] {
        let mut layout = Layout::logical_linear(1, 1);
        if shape.len() > 1 && shape[0] % u32::from(tiles) == 0 && rng.bool() {
            layout = Layout::row_sharded(tiles);
        }
        if rng.bool() {
            layout.memory_class = MemoryClass::Ipu21Interleaved;
        }
        config.inputs.insert(
            input,
            TensorFormat {
                precision: Precision::F16,
                layout,
            },
        );
    }
    let mut choices = boundary_layouts(&high, &config);
    if rng.bool() && !high.outputs().contains(&weight) {
        config.inputs.remove(&weight);
        choices.insert(weight, None);
    }
    for op in high.operations() {
        if rng.usize(0..4) == 0 {
            choices.insert(op.results[0], Some(config.inputs[&x].layout.clone()));
        }
    }
    (high, config, choices)
}

#[test]
fn randomized_dp_matches_complete_path_enumeration() {
    for seed in 0..128 {
        let mut rng = fastrand::Rng::with_seed(0xd15e_a5e0 + seed);
        let (high, mut config, choices) = random_fixture(&mut rng);
        let complete = enumerate(&high, &choices, &config, &[]);
        assert!(!complete.is_empty(), "seed={seed}");
        let minimum = complete.iter().map(|g| g.peak_memory.total).min().unwrap();
        let maximum = complete.iter().map(|g| g.peak_memory.total).max().unwrap();
        let budgets = BTreeSet::from([minimum - 1, minimum, rng.u64(minimum..=maximum), maximum]);
        let mut previous = None;
        for budget in budgets {
            config.tile_memory_budget_bytes = budget;
            let expected = optimum(&complete, &config);
            let selected = plan(&high, &choices, &config, EXACT);
            assert_eq!(
                result_cycles(&selected),
                expected,
                "seed={seed}, budget={budget}, high={high:?}"
            );
            let reversed = explore(&high, &choices, &config, &[], EXACT, true).unwrap();
            check_frontier(&complete, &reversed, seed);
            assert_eq!(
                result_cycles(&reversed.finish()),
                expected,
                "enumeration-order dependence: seed={seed}, budget={budget}"
            );
            if let Ok(mid) = selected {
                if let Some(previous) = previous {
                    assert!(
                        mid.estimated_cycles <= previous,
                        "budget monotonicity: seed={seed}"
                    );
                }
                previous = Some(mid.estimated_cycles);
                check_semantics(&high, &mid, &mut rng);
                mid.validate().unwrap();
                crate::low::expand::expand_tiles(Target::Ipu21, &mid, false)
                    .unwrap_or_else(|e| panic!("seed={seed}, budget={budget}: {e:?}"));
                for (&value, layout) in &choices {
                    if let Some(layout) = layout {
                        if let Some(represented) = mid.values.iter().find(|v| v.origin == value) {
                            assert_eq!(
                                &represented.tensor_type.format.layout, layout,
                                "seed={seed}"
                            );
                        }
                    }
                }
            }
            let beam = explore(
                &high,
                &choices,
                &config,
                &[],
                SearchLimits {
                    states_per_boundary: Some(2),
                    paths_per_state: Some(1),
                },
                false,
            )
            .and_then(Search::finish);
            if let Ok(mid) = beam {
                assert!(
                    mid.estimated_cycles >= expected.unwrap(),
                    "beam beat exhaustive oracle: seed={seed}"
                );
                assert!(mid.peak_memory.fits_with_budget(config.target, 0, budget));
                check_semantics(&high, &mid, &mut rng);
            }
        }
    }
}

#[test]
fn randomized_memory_tradeoffs_match_exhaustive_paths() {
    for seed in 0..64 {
        let mut rng = fastrand::Rng::with_seed(0xbad6_e700 + seed);
        let mut high = HighGraph::new();
        let x = high.host_input("x", [4, 16]).unwrap();
        let mut output = x;
        for _ in 0..rng.usize(2..=4) {
            output = high.gelu(output).unwrap();
        }
        high.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(Target::Ipu21, 1).with_input(
            x,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            },
        );
        config.standard_memory_reservation_bytes = 0;
        let mut choices = boundary_layouts(&high, &config);
        for op in high.operations() {
            choices.insert(op.results[0], Some(Layout::row_sharded(1)));
        }
        let large = rng.u32(256..=1024) * 8;
        let small = 64;
        let scratch = [
            (large, 0, MemoryClass::Ipu21Standard),
            (small, rng.usize(1..=3), MemoryClass::Ipu21Standard),
            (large / 2, 1, MemoryClass::Ipu21Interleaved),
        ];
        let complete = enumerate(&high, &choices, &config, &scratch);
        let fastest = complete.iter().min_by_key(|g| g.estimated_cycles).unwrap();
        let smallest = complete.iter().min_by_key(|g| g.peak_memory.total).unwrap();
        assert!(
            fastest.estimated_cycles < smallest.estimated_cycles
                && fastest.peak_memory.total > smallest.peak_memory.total,
            "fixture failed to exercise a real tradeoff: seed={seed}"
        );
        for budget in [
            smallest.peak_memory.total - 1,
            smallest.peak_memory.total,
            fastest.peak_memory.total,
        ] {
            config.tile_memory_budget_bytes = budget;
            for reverse in [false, true] {
                let search = explore(&high, &choices, &config, &scratch, EXACT, reverse).unwrap();
                check_frontier(&complete, &search, seed);
                let selected = search.finish();
                assert_eq!(
                    result_cycles(&selected),
                    optimum(&complete, &config),
                    "seed={seed}, budget={budget}"
                );
                if let Ok(mid) = selected {
                    check_semantics(&high, &mid, &mut rng);
                }
            }
        }
    }
}

#[test]
fn interleaved_capacity_is_enforced_even_when_total_storage_fits() {
    use ipu_target::ipu21::memory::{IPU21_INTERLEAVED_REGION_BYTES, IPU21_PLANNED_DATA_BYTES};
    for seed in 0..8 {
        let mut rng = fastrand::Rng::with_seed(0xc1a5_5000 + seed);
        let mut high = HighGraph::new();
        let x = high.host_input("x", [4, 16]).unwrap();
        let y = high.gelu(x).unwrap();
        high.set_outputs([y]).unwrap();
        let mut config = PipelineConfig::new(Target::Ipu21, 1).with_input(
            x,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(1),
            },
        );
        config.standard_memory_reservation_bytes = 0;
        config.tile_memory_budget_bytes = u64::from(IPU21_PLANNED_DATA_BYTES);
        let choices = boundary_layouts(&high, &config);
        let scratch = [
            (
                IPU21_INTERLEAVED_REGION_BYTES + rng.u32(1..=128) * 8,
                0,
                MemoryClass::Ipu21Interleaved,
            ),
            (64, rng.usize(1..=3), MemoryClass::Ipu21Standard),
        ];
        let complete = enumerate(&high, &choices, &config, &scratch);
        let fastest = complete.iter().min_by_key(|g| g.estimated_cycles).unwrap();
        assert!(fastest.peak_memory.total < config.tile_memory_budget_bytes);
        assert!(fastest.peak_memory.interleaved > u64::from(IPU21_INTERLEAVED_REGION_BYTES));
        let search = explore(&high, &choices, &config, &scratch, EXACT, rng.bool()).unwrap();
        check_frontier(&complete, &search, seed);
        let selected = search.finish().unwrap();
        assert_eq!(Some(selected.estimated_cycles), optimum(&complete, &config));
        assert!(selected.estimated_cycles > fastest.estimated_cycles);
    }
}

#[test]
fn late_resident_choices_match_whole_program_memory_under_tight_budgets() {
    for seed in 0..32 {
        let mut rng = fastrand::Rng::with_seed(0x1a7e_0000 + seed);
        let mut high = HighGraph::new();
        let rows = 4 * rng.u32(1..=4);
        let x = high.host_input("x", [rows, 16]).unwrap();
        let a = high.parameter("a", [rows, 16]).unwrap();
        let b = high.parameter("b", [rows, 16]).unwrap();
        let first = high.gelu(x).unwrap();
        let second = high.add(first, a).unwrap();
        let third = high.add(second, b).unwrap();
        let last = high.add(third, a).unwrap();
        high.set_outputs([last]).unwrap();
        let mut config = PipelineConfig::new(Target::Ipu21, 4).with_input(
            x,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_sharded(4),
            },
        );
        config.standard_memory_reservation_bytes = 0;
        let mut choices = boundary_layouts(&high, &config);
        for op in high.operations() {
            choices.insert(op.results[0], Some(Layout::row_sharded(4)));
        }
        let scratch = [
            (rng.u32(32..=128) * 8, 0, MemoryClass::Ipu21Standard),
            (64, 1, MemoryClass::Ipu21Interleaved),
        ];
        let complete = enumerate(&high, &choices, &config, &scratch);
        let minimum = complete.iter().map(|g| g.peak_memory.total).min().unwrap();
        let maximum = complete.iter().map(|g| g.peak_memory.total).max().unwrap();
        for budget in [minimum - 1, minimum, maximum] {
            config.tile_memory_budget_bytes = budget;
            let complete = enumerate(&high, &choices, &config, &scratch);
            let expected = optimum(&complete, &config);
            let search = explore(&high, &choices, &config, &scratch, EXACT, rng.bool()).unwrap();
            let selected = search.finish();
            assert_eq!(
                result_cycles(&selected),
                expected,
                "seed={seed}, budget={budget}"
            );
            if let Ok(graph) = selected {
                check_semantics(&high, &graph, &mut rng);
                // Both parameters are initial, permanent inputs, including a
                // shared parameter whose consumers are separated by b's use.
                assert_eq!(
                    graph
                        .inputs
                        .iter()
                        .filter(|i| i.kind == GraphInputKind::Parameter)
                        .count(),
                    2
                );
                graph.validate().unwrap();
            }
        }
    }
}
