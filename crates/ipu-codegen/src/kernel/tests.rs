use super::*;
use crate::estimate::Ipu21CostModel;
use crate::mid::MidOperationKind;
use crate::planner::test_support::lower;
use crate::{
    AccumulationPrecision, ComputeGraph, KernelRequirements, Layout, MemoryClass, PipelineConfig,
    ShardExtent, ShardView, TensorFormat, TensorTiling, WorkProvenance, WorkReason, lower_to_tiles,
};

#[test]
fn column_sharded_add_partitions_multi_row_broadcast_parameters() {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [8, 4, 32]).unwrap();
    let bias = graph.host_input("bias", [1, 4, 32]).unwrap();
    let output = graph.add(x, bias).unwrap();
    graph.set_outputs([output]).unwrap();
    let format = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_major(TensorTiling::sharded(crate::TensorAxis::FromEnd(1), 4)),
    };
    let config = PipelineConfig::new(4)
        .with_input(x, format.clone())
        .with_input(bias, format);
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
    let addresses = low
        .shards
        .iter()
        .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
        .collect();
    for run in &low.kernel_runs {
        if matches!(run.kernel, MidOperationKind::Add) {
            materialize_kernel_run(run, &low.shards, &addresses, &BTreeMap::new()).unwrap();
            assert_eq!(
                run.inputs[1].extents.last().unwrap().physical_end
                    - run.inputs[1].extents.last().unwrap().start,
                8
            );
        }
    }
}

#[test]
fn packed_add_keeps_padding_in_dense_operand_view() {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [2, 3, 32]).unwrap();
    let y = graph.host_input("y", [2, 3, 32]).unwrap();
    let output = graph.add(x, y).unwrap();
    graph.set_outputs([output]).unwrap();
    let mut layout = Layout::amp_left(16, 2);
    layout.tiling.axes[0] =
        crate::AxisTiling::new(crate::TensorAxis::FromEnd(2), 2, 1, crate::Padding::Zero)
            .with_shard_padding_multiple(4);
    let format = TensorFormat {
        precision: Precision::F16,
        layout,
    };
    let config = PipelineConfig::new(2)
        .with_input(x, format.clone())
        .with_input(y, format);
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
    let addresses = low
        .shards
        .iter()
        .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
        .collect();
    let mut packed_adds = 0;
    for run in &low.kernel_runs {
        if matches!(run.kernel, MidOperationKind::Add) {
            assert_eq!(
                run.requirements.outputs[0].layout.order,
                ElementOrder::Amp(AmpOrder::Left)
            );
            packed_adds += 1;
            materialize_kernel_run(run, &low.shards, &addresses, &BTreeMap::new()).unwrap();
        }
    }
    assert_eq!(packed_adds, 2);
}

#[test]
fn fp8_gemms_repack_casts_and_keep_half_outputs() {
    let fp8 = Precision::F8F143 { scale_exponent: -4 };
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [129, 128]).unwrap();
    let weights = graph.parameter("weights", [128, 64]).unwrap();
    let output = graph.gemm(input, weights).unwrap();
    graph.set_outputs([output]).unwrap();
    let mut config = PipelineConfig::new(64)
        .with_input(
            input,
            TensorFormat {
                precision: Precision::F16,
                layout: Layout::amp_left(64, 64),
            },
        )
        .with_automatic_input(weights, fp8);
    config.operator_candidates = vec![crate::planner::OperatorCandidate::fp8_gemm(64, -4)];
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
    let plan = KernelObjects::from_program(&low).unwrap();
    assert!(plan.compilations.iter().any(|compilation| {
        compilation
            .flags
            .iter()
            .any(|flag| flag == "-DGEMM_NATIVE_FP8=1")
    }));
    let mut casts = 0;
    let mut gemms = 0;
    for tile in &low.tiles {
        for work in tile.work.iter() {
            let BlockOperation::Compute { run, .. } = work else {
                continue;
            };
            let run = &low.kernel_runs[run.0 as usize];
            match run.kernel {
                MidOperationKind::Cast { to, .. } if to == fp8 => {
                    casts += 1;
                    assert_eq!(
                        run.requirements.inputs[0].layout.order,
                        run.requirements.outputs[0].layout.order
                    );
                    assert_ne!(
                        run.requirements.outputs[0].layout.order,
                        ElementOrder::RowMajor
                    );
                    let abi = run.call(None).unwrap();
                    assert!(abi.arguments[3] > 0);
                }
                MidOperationKind::Gemm {
                    multiply,
                    accumulate,
                    ..
                } => {
                    gemms += 1;
                    assert_eq!(multiply, fp8);
                    assert_eq!(accumulate, AccumulationPrecision::F16);
                    assert_eq!(run.requirements.outputs[0].precision, Precision::F16);
                    assert_eq!(run.call(None).unwrap().arguments, vec![(-8i32) as u32]);
                    let mut rescaled = run.clone();
                    let metadata = std::sync::Arc::make_mut(&mut rescaled.metadata);
                    if let MidOperationKind::Gemm { multiply, .. } = &mut metadata.kernel {
                        *multiply = Precision::F8F143 { scale_exponent: 1 };
                    }
                    for input in &mut metadata.requirements.inputs {
                        input.precision = Precision::F8F143 { scale_exponent: 1 };
                    }
                    assert_eq!(
                        run.call(None).unwrap().symbol,
                        rescaled.call(None).unwrap().symbol
                    );
                    assert_eq!(rescaled.call(None).unwrap().arguments, vec![2]);
                }
                _ => {}
            }
        }
    }
    assert!(casts > 0 && gemms > 0);
}

#[test]
fn randomized_gemm_row_specializations_follow_physical_output_orientation() {
    let mut random = fastrand::Rng::with_seed(0x726f_7773_6f72_6465);
    for case in 0..256 {
        let outer = random.u32(1..=4);
        let semantic_rows = random.u32(1..=96);
        let semantic_columns = random.u32(1..=96);
        let transposed = random.bool();
        let order = match (transposed, random.bool()) {
            (false, false) => AmpOrder::Output,
            (false, true) => AmpOrder::Left,
            (true, false) => AmpOrder::TransposedOutput,
            (true, true) => AmpOrder::TransposedLeft,
        };
        let format = TensorFormat {
            precision: Precision::F16,
            layout: Layout {
                order: ElementOrder::Amp(order),
                tiling: TensorTiling::replicated(1),
                memory_class: MemoryClass::Ipu21Standard,
            },
        };
        let run = KernelRun::new(
            WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::OperatorKernel,
            },
            MidOperationKind::Gemm {
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
                mode: GemmKernelMode::Initialize,
                weights: GemmWeightLoad::Standard,
                inner_block: 64,
                output_columns: 16,
                axes: crate::GemmAxes {
                    left_inner: crate::TensorAxis::FromEnd(1),
                    right_inner: crate::TensorAxis::FromEnd(2),
                    output_column: crate::TensorAxis::FromEnd(1),
                    valid_inner: None,
                    valid_columns: None,
                },
            },
            Vec::new(),
            vec![ShardView {
                shard: BlockValueId::from_index(0),
                extents: [outer, semantic_rows, semantic_columns]
                    .into_iter()
                    .enumerate()
                    .map(|(axis, physical_end)| ShardExtent {
                        axis: axis as u16,
                        start: 0,
                        logical_end: physical_end,
                        physical_end,
                    })
                    .collect(),
            }],
            KernelRequirements {
                inputs: Vec::new(),
                outputs: vec![format],
                distinct_elements: Vec::new(),
            },
        );
        let expected = outer
            * if transposed {
                semantic_columns
            } else {
                semantic_rows
            };
        assert_eq!(
            gemm_rows(run.geometry(MemoryOperand::Output(0))).unwrap(),
            expected,
            "random case {case}"
        );
    }
}

#[test]
fn randomized_gemm_plans_compile_and_select_scheduled_row_specializations() {
    let mut random = fastrand::Rng::with_seed(0x7370_6563);
    for _ in 0..32 {
        let tiles = 1_u16 << random.u32(0..=3);
        let rows_per_tile = random.u32(1..=12);
        let batch = random.u32(1..=4);
        let mut graph = ComputeGraph::new();
        let left = graph
            .host_input("left", [batch, u32::from(tiles) * rows_per_tile, 64])
            .unwrap();
        let right = graph.parameter("right", [64, 64]).unwrap();
        let result = graph.gemm(left, right).unwrap();
        graph.set_outputs([result]).unwrap();
        let mut config = PipelineConfig::new(tiles)
            .with_active_tile_counts([tiles])
            .with_input(
                left,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::amp_left(64, tiles),
                },
            )
            .with_input(
                right,
                TensorFormat {
                    precision: Precision::F16,
                    layout: Layout::block_major_matrix(64, tiles),
                },
            );
        // This fixture exercises row specialization of a single GEMM family,
        // independently of changes to the planner's relative strategy costs.
        config.operator_candidates.retain(|candidate| {
            matches!(candidate,
            crate::planner::OperatorCandidate::Concrete(candidate) if matches!(candidate.plan.dispatch,
                crate::planner::operator::OperatorDispatch::BlockedGemm { orientation: crate::planner::operator::GemmOrientation::Normal,
                    distribution: crate::planner::operator::GemmDistribution::OutputStationary, .. }))
        });
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(
            &crate::expand_tiles(&mid).unwrap(),
            config.diagnostic_checkpoints,
        );
        let plan = KernelObjects::from_program(&low).unwrap();
        let addresses = low
            .shards
            .iter()
            .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
            .collect::<BTreeMap<_, _>>();
        let specialization = plan
            .compilations
            .iter()
            .find(|unit| {
                unit.flags
                    .iter()
                    .any(|flag| flag.starts_with("-DGEMM_SMALL_ROWS="))
            })
            .unwrap();
        assert_eq!(plan.compilations.len(), 3);
        assert_eq!(
            plan.compilations
                .iter()
                .filter(|unit| unit.name.ends_with("_dispatch"))
                .count(),
            1
        );
        let planned_rows = low
            .kernel_runs
            .iter()
            .filter(|run| matches!(run.kernel, MidOperationKind::Gemm { .. }))
            .map(|run| gemm_rows(run.geometry(MemoryOperand::Output(0))).unwrap())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert!(
            specialization
                .flags
                .iter()
                .any(|flag| flag == &format!("-DGEMM_SMALL_ROWS={}", planned_rows[0]))
        );
        assert!(
            specialization
                .flags
                .iter()
                .any(|flag| flag == "-DGEMM_SINGLE_ROWS=1")
        );
        assert!(
            plan.retained_symbols()
                .all(|symbol| !symbol.contains("accumulate"))
        );
        for run in low
            .tiles
            .iter()
            .flat_map(|tile| tile.work.iter())
            .filter_map(|work| {
                if let BlockOperation::Compute { run, .. } = work {
                    Some(&low.kernel_runs[run.0 as usize])
                } else {
                    None
                }
            })
        {
            let call = run.call(None).unwrap();
            assert!(
                plan.retained_symbols()
                    .any(|symbol| symbol == call.symbol.as_str())
            );
            assert!(call.arguments.is_empty());
            let compute = materialize_kernel_run(run, &low.shards, &addresses, &BTreeMap::new())
                .unwrap_or_else(|error| panic!("batch={batch} tiles={tiles} run={run:?}: {error}"));
            assert_eq!(compute.symbol, call.symbol.as_str());
            assert_eq!(compute.input_addresses.len(), 2);
        }
    }
}

#[test]
fn attention_stages_support_multiple_configurations_and_block_sizes() {
    let mut plan = KernelObjects::default();
    let mut stages = Vec::new();
    for head in [32, 64] {
        for padded in [32, 64] {
            for rows in [1, 3, 7] {
                for keys in [1, padded / 2, padded] {
                    stages.push((
                        MidOperationKind::AttentionSoftmax {
                            head_dimension: head,
                            key_columns: keys,
                            padded_key_columns: padded,
                        },
                        rows,
                    ));
                }
            }
        }
    }
    for values in [16, 32] {
        for rows in [1, 3, 7] {
            stages.push((
                MidOperationKind::AttentionMerge {
                    value_dimension: values,
                    padded_value_dimension: values,
                    initial: true,
                    final_block: false,
                },
                rows,
            ));
        }
    }
    let mut calls = Vec::new();
    for (kernel, rows) in stages {
        let (inputs, outputs, expected) = match kernel {
            MidOperationKind::AttentionSoftmax {
                key_columns,
                padded_key_columns,
                ..
            } => {
                let probability = crate::TensorType::new(
                    [rows, padded_key_columns],
                    Precision::F16,
                    Layout::amp_left(16, 1),
                );
                let mut outputs = vec![probability.clone()];
                outputs.extend(
                    softmax_workspaces(&probability, key_columns != padded_key_columns).unwrap(),
                );
                (vec![probability], outputs, vec![rows, key_columns, 0])
            }
            MidOperationKind::AttentionMerge {
                value_dimension, ..
            } => (
                vec![
                    crate::TensorType::new(
                        [rows, value_dimension],
                        Precision::F16,
                        Layout::amp_left(16, 1),
                    ),
                    crate::TensorType::new([2, rows], Precision::F32, Layout::row_sharded(1)),
                ],
                vec![crate::TensorType::new(
                    [rows, value_dimension + 2],
                    Precision::F32,
                    Layout {
                        order: ElementOrder::RowMajor,
                        ..Layout::amp_left(16, 1)
                    },
                )],
                vec![1, 0, rows],
            ),
            _ => unreachable!(),
        };
        let input_count = inputs.len();
        let shards = inputs
            .into_iter()
            .chain(outputs)
            .enumerate()
            .map(|(index, tensor_type)| BlockValue {
                id: BlockValueId::from_index(index as u32),
                tile: 0,
                extents: tensor_type
                    .format
                    .layout
                    .shard_extents(&tensor_type.shape)
                    .unwrap()[0]
                    .1
                    .clone(),
                tensor_type,
                definition: crate::ShardDefinition::Staging,
            })
            .collect::<Vec<_>>();
        let views = shards
            .iter()
            .map(|shard| ShardView {
                shard: shard.id,
                extents: shard.extents.clone(),
            })
            .collect::<Vec<_>>();
        let run = KernelRun::bind(
            WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::OperatorKernel,
            },
            kernel,
            views[..input_count].to_vec(),
            views[input_count..].to_vec(),
            &shards,
            &mut Vec::new(),
        )
        .unwrap();
        let call = run.call(Some(&mut plan)).unwrap();
        assert_eq!(call.arguments, expected);
        calls.push(call);
    }
    // Four dimension pairs each have full/masked softmax, plus two merges.
    // Query-row counts share assembly instead of producing more codelets.
    assert_eq!(plan.compilations.len(), 10);
    for call in calls {
        assert!(
            plan.retained_symbols()
                .any(|symbol| symbol == call.symbol.as_str())
        );
    }
}

#[test]
fn block_rearrangements_have_distinct_objects_and_symbols() {
    // These layouts have identical matrix sizes and C++ order indices, but
    // require different worker code. They must coexist in one linked package.
    let targets = [(8, 16), (16, 8), (64, 16)];
    let mut plan = KernelObjects::default();
    let mut calls = Vec::new();
    for (row_block, column_block) in targets {
        calls.push(packing_call(
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block,
                column_block,
            }),
            128,
            128,
            128,
            128,
            Some(&mut plan),
        ));
    }
    let objects = plan
        .compilations
        .iter()
        .map(|object| &object.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(objects.len(), plan.compilations.len());
    let symbols = plan.retained_symbols().collect::<BTreeSet<_>>();
    assert_eq!(symbols.len(), targets.len());
    for ((row_block, column_block), call) in targets.into_iter().zip(calls) {
        let symbol = &call.symbol;
        assert!(symbols.contains(symbol.as_str()));
        let object = plan
            .compilations
            .iter()
            .find(|object| {
                object.source == "rearrange_f16.cpp" && object.name == format!("{symbol}_codelet")
            })
            .unwrap();
        assert!(
            object
                .flags
                .contains(&format!("-DREARRANGE_ROW_BLOCK={row_block}"))
        );
        assert!(
            object
                .flags
                .contains(&format!("-DREARRANGE_COLUMN_BLOCK={column_block}"))
        );
    }
}

#[test]
fn zero_ranges_use_range_arguments_and_stay_inside_the_output_view() {
    let tensor_type = crate::TensorType::new(
        [128],
        Precision::F16,
        Layout::row_major(TensorTiling::replicated(1)),
    );
    let shard = BlockValue {
        id: BlockValueId::from_index(0),
        tile: 0,
        tensor_type: tensor_type.clone(),
        extents: vec![ShardExtent {
            axis: 0,
            start: 0,
            logical_end: 128,
            physical_end: 128,
        }],
        definition: crate::ShardDefinition::Staging,
    };
    let run = KernelRun::new(
        WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::LayoutRearrangement,
        },
        MidOperationKind::FillZero {
            offset: 16,
            bytes: 56,
            padding_only: false,
        },
        Vec::new(),
        vec![ShardView {
            shard: shard.id,
            extents: vec![ShardExtent {
                axis: 0,
                start: 16,
                logical_end: 64,
                physical_end: 64,
            }],
        }],
        KernelRequirements {
            inputs: Vec::new(),
            outputs: vec![tensor_type.format],
            distinct_elements: Vec::new(),
        },
    );
    let addresses = BTreeMap::from([(shard.id, 0x60000)]);
    let materialize = |run: &KernelRun| {
        materialize_kernel_run(
            run,
            std::slice::from_ref(&shard),
            &addresses,
            &BTreeMap::new(),
        )
    };
    let compute = materialize(&run).unwrap();
    assert_eq!(
        compute.output_address,
        TileAddress::Absolute(0x60000 + 32 + 16)
    );
    assert_eq!(run.call(None).unwrap().arguments, vec![1, 1]);
    for (offset, bytes) in [(48, 56), (1, 8), (0, 7), (u32::MAX - 7, 16)] {
        let run = KernelRun::new(
            run.provenance,
            MidOperationKind::FillZero {
                offset,
                bytes,
                padding_only: false,
            },
            run.inputs.clone(),
            vec![run.outputs[0].clone()],
            run.requirements.clone(),
        );
        assert!(materialize(&run).is_err(), "offset={offset} bytes={bytes}");
    }
}

#[test]
fn packed_gemm_stores_bind_without_output_copies() {
    let orientation = crate::planner::operator::GemmOrientation::Normal;
    {
        for rows in [17, 96, 129] {
            let mut graph = ComputeGraph::new();
            let left = graph.host_input("left", [1, rows, 64]).unwrap();
            let right = graph.parameter("right", [64, 80]).unwrap();
            let output = graph.gemm(left, right).unwrap();
            graph.set_outputs([output]).unwrap();
            let format = TensorFormat {
                precision: Precision::F16,
                layout: Layout::row_major(TensorTiling::replicated(1)),
            };
            let mut config = PipelineConfig::new(1)
                .with_input(left, format.clone())
                .with_input(right, format);
            config.gemm_output_packing = crate::GemmOutputPacking::Packed;
            config.operator_candidates.retain(|candidate| matches!(candidate,
                crate::planner::OperatorCandidate::Concrete(candidate) if matches!(candidate.plan.dispatch,
                    crate::planner::operator::OperatorDispatch::BlockedGemm { orientation: candidate_orientation,
                        distribution: crate::planner::operator::GemmDistribution::OutputStationary, .. } if candidate_orientation == orientation)));
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
            let addresses = low
                .shards
                .iter()
                .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
                .collect();
            let mut products = 0;
            for run in &low.kernel_runs {
                if !matches!(run.kernel, MidOperationKind::Gemm { .. }) {
                    continue;
                }
                products += 1;
                assert!(
                    gemm_rows(run.geometry(MemoryOperand::Output(0)))
                        .unwrap()
                        .is_multiple_of(16)
                );
                assert_eq!(
                    run.requirements.outputs[0].layout.order.gemm_output_group(),
                    Some(64)
                );
                let step =
                    materialize_kernel_run(run, &low.shards, &addresses, &BTreeMap::new()).unwrap();
                assert!(step.symbol.contains("packed64"));
                let source = &low.shards[run.outputs[0].shard.index() as usize];
                assert_eq!(
                    source.tensor_type.format.layout.order,
                    run.requirements.outputs[0].layout.order
                );
            }
            assert!(products > 0);
        }
    }
}

#[test]
fn f32_to_f16_cast_calls_cover_partial_worker_waves() {
    let mut plan = KernelObjects::default();
    for count in [1, 2, 11, 12, 13, 72, 729, 1152] {
        let format = |precision| TensorFormat {
            precision,
            layout: Layout::row_major(TensorTiling::replicated(1)),
        };
        let view = |id| ShardView {
            shard: BlockValueId::from_index(id),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: count,
                physical_end: count,
            }],
        };
        let run = KernelRun::new(
            WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::PrecisionCast,
            },
            MidOperationKind::Cast {
                from: Precision::F32,
                to: Precision::F16,
            },
            vec![view(0)],
            vec![view(1)],
            KernelRequirements {
                inputs: vec![format(Precision::F32)],
                outputs: vec![format(Precision::F16)],
                distinct_elements: Vec::new(),
            },
        );
        let call = run.call(Some(&mut plan)).unwrap();
        assert_eq!(call.arguments, [count]);
        assert!(
            plan.retained_symbols()
                .any(|symbol| symbol == call.symbol.as_str())
        );
    }
}

#[test]
fn shared_row_tails_preserve_column_alignment_for_wide_packing() {
    // 14 columns have a two-halfword tail: a 64-bit load would cross the row.
    // Sharing kernels across row tails must not erase this distinction.
    for columns in [14, 16] {
        let mut plan = KernelObjects::default();
        packing_call(
            ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
                row_block: 64,
                column_block: 16,
            }),
            63,
            64,
            columns,
            16,
            Some(&mut plan),
        );
        let source = plan
            .compilations
            .iter()
            .find(|object| object.source.starts_with("rearrange_"))
            .unwrap()
            .source;
        assert_eq!(
            source,
            if columns == 14 {
                "rearrange_f16.cpp"
            } else {
                "rearrange_block_major_f16.S"
            }
        );
    }
}

#[test]
fn unsupported_kernel_formats_fail_at_call_construction() {
    let format = TensorFormat {
        precision: Precision::F32,
        layout: Layout::row_sharded(1),
    };
    for (kernel, inputs) in [
        (MidOperationKind::Add, 2),
        (MidOperationKind::LayerNorm, 3),
        (MidOperationKind::Gelu, 1),
        (
            MidOperationKind::Cast {
                from: Precision::F16,
                to: Precision::F32,
            },
            1,
        ),
        (
            MidOperationKind::Rearrange {
                from: format.layout.clone(),
                to: format.layout.clone(),
            },
            1,
        ),
    ] {
        let requirements = KernelRequirements {
            inputs: vec![format.clone(); inputs],
            outputs: vec![format.clone()],
            distinct_elements: vec![],
        };
        let view = ShardView {
            shard: BlockValueId(0),
            extents: vec![ShardExtent {
                axis: 0,
                start: 0,
                logical_end: 16,
                physical_end: 16,
            }],
        };
        let run = KernelRun::new(
            WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::OperatorKernel,
            },
            kernel.clone(),
            vec![view.clone(); inputs],
            vec![view],
            requirements,
        );
        assert_eq!(run.call(None), Err(KernelError::Unavailable(kernel)));
    }
}

#[test]
fn bias_gelu_rejects_broadcast_volume_overflow() {
    let format = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(1),
    };
    let view = |shard, sizes: [u32; 2]| ShardView {
        shard: BlockValueId(shard),
        extents: sizes
            .into_iter()
            .enumerate()
            .map(|(axis, size)| ShardExtent {
                axis: axis as u16,
                start: 0,
                logical_end: size,
                physical_end: size,
            })
            .collect(),
    };
    let kernel = MidOperationKind::BiasGelu;
    let mut run = KernelRun::new(
        WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::OperatorKernel,
        },
        kernel.clone(),
        vec![view(0, [1, 2]), view(1, [1, 2])],
        vec![view(2, [1, 2])],
        KernelRequirements {
            inputs: vec![format.clone(); 2],
            outputs: vec![format],
            distinct_elements: vec![],
        },
    );
    run.call(None).unwrap();
    // An unchecked u32 product wraps to the expected bias width of two.
    run.inputs[1] = view(1, [2, (1 << 31) + 1]);
    assert_eq!(run.call(None), Err(KernelError::ElementCountOverflow));
}

#[test]
fn binding_checks_backing_strides_before_placement() {
    let provenance = WorkProvenance {
        operation: None,
        value: None,
        reason: WorkReason::OperatorKernel,
    };
    let tensor = crate::TensorType::new([4, 16], Precision::F16, Layout::row_sharded(1));
    let extents = tensor.format.layout.shard_extents(&tensor.shape).unwrap()[0]
        .1
        .clone();
    let shards = (0..2)
        .map(|index| BlockValue {
            id: BlockValueId(index),
            tile: 0,
            tensor_type: tensor.clone(),
            extents: extents.clone(),
            definition: crate::ShardDefinition::Staging,
        })
        .collect::<Vec<_>>();
    for rows in 1..=4 {
        for columns in [4, 8, 16] {
            let views = shards
                .iter()
                .map(|shard| {
                    let mut extents = shard.extents.clone();
                    extents[0].logical_end = rows;
                    extents[0].physical_end = rows;
                    extents[1].logical_end = columns;
                    extents[1].physical_end = columns;
                    ShardView {
                        shard: shard.id,
                        extents,
                    }
                })
                .collect::<Vec<_>>();
            let bound = KernelRun::bind(
                provenance,
                MidOperationKind::Gelu,
                vec![views[0].clone()],
                vec![views[1].clone()],
                &shards,
                &mut Vec::new(),
            );
            // Slicing columns retains the backing row stride. It is dense
            // only for a single row or for complete rows of that storage.
            if rows == 1 || columns == 16 {
                let run = bound.unwrap();
                let addresses = BTreeMap::from([(shards[0].id, 0x60000), (shards[1].id, 0x70000)]);
                materialize_kernel_run(&run, &shards, &addresses, &BTreeMap::new()).unwrap();
            } else {
                assert!(matches!(bound, Err(KernelError::FragmentedView { .. })));
            }
        }
    }
    let mut output = tensor.format;
    output.layout.order = ElementOrder::Amp(AmpOrder::Left);
    let mut incompatible = shards.clone();
    incompatible[1].tensor_type.format = output;
    assert!(matches!(
        KernelRun::bind(
            provenance,
            MidOperationKind::Gelu,
            vec![ShardView {
                shard: shards[0].id,
                extents: extents.clone()
            }],
            vec![ShardView {
                shard: shards[1].id,
                extents
            }],
            &incompatible,
            &mut Vec::new(),
        ),
        Err(KernelError::Unavailable(MidOperationKind::Gelu))
    ));
}
#[test]
fn object_registration_reuses_identical_definitions_and_rejects_conflicts() {
    let mut plan = KernelObjects::default();
    let unit = KernelCompilation {
        source: "worker_support.S",
        name: "worker_support".into(),
        flags: Vec::new(),
    };
    plan.add_compilation(unit.clone());
    plan.add_compilation(unit.clone());
    assert_eq!(plan.compilations, vec![unit.clone()]);
    for conflicting in [
        KernelCompilation {
            source: "worker_call.S",
            ..unit.clone()
        },
        KernelCompilation {
            flags: vec!["-DOTHER=1".into()],
            ..unit.clone()
        },
    ] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plan.add_compilation(conflicting);
            }))
            .is_err()
        );
        assert_eq!(plan.compilations, vec![unit.clone()]);
    }
}

fn packing_call(
    order: ElementOrder,
    rows: u32,
    physical_rows: u32,
    columns: u32,
    physical_columns: u32,
    build: Option<&mut KernelObjects>,
) -> KernelCall {
    let source = TensorFormat {
        precision: Precision::F16,
        layout: Layout::row_sharded(1),
    };
    let mut target = source.clone();
    target.layout.order = order;
    let extents = [(rows, physical_rows), (columns, physical_columns)]
        .into_iter()
        .enumerate()
        .map(|(axis, (logical_end, physical_end))| ShardExtent {
            axis: axis as u16,
            start: 0,
            logical_end,
            physical_end,
        })
        .collect::<Vec<_>>();
    KernelCall::select(
        &MidOperationKind::Rearrange {
            from: source.layout.clone(),
            to: target.layout.clone(),
        },
        &[TensorStorage {
            format: &source,
            extents: &extents,
        }],
        &[TensorStorage {
            format: &target,
            extents: &extents,
        }],
        build,
    )
    .unwrap()
}
