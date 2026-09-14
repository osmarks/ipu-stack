use super::*;
use crate::{
    AccumulationPrecision, ComputeGraph, Ipu21CostModel, KernelAccess, KernelRequirements, Layout,
    MemoryClass, PipelineConfig, ShardExtent, ShardView, TensorFormat, TensorTiling,
    WorkProvenance, WorkReason, lower, lower_to_tiles,
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
    let build = KernelBuildPlan::from_program(&low).unwrap();
    let addresses = low
        .shards
        .iter()
        .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
        .collect();
    for run in &low.kernel_runs {
        if matches!(run.kernel, TileKernelSpec::Add) {
            materialize_kernel_run(run, &low.shards, &addresses, &build, &BTreeMap::new()).unwrap();
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
    let build = KernelBuildPlan::from_program(&low).unwrap();
    let addresses = low
        .shards
        .iter()
        .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
        .collect();
    let mut packed_adds = 0;
    for run in &low.kernel_runs {
        if matches!(run.kernel, TileKernelSpec::Add) {
            assert_eq!(
                run.requirements.outputs[0].format.layout.order,
                ElementOrder::Amp(AmpOrder::Left)
            );
            packed_adds += 1;
            materialize_kernel_run(run, &low.shards, &addresses, &build, &BTreeMap::new()).unwrap();
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
    config.operator_candidates = vec![crate::OperatorCandidate::fp8_gemm(64, -4)];
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
    let plan = KernelBuildPlan::from_program(&low).unwrap();
    assert!(plan.compilations.iter().any(|compilation| {
        compilation
            .flags
            .iter()
            .any(|flag| flag == "-DGEMM_NATIVE_FP8=1")
    }));
    let mut casts = 0;
    let mut gemms = 0;
    for tile in &low.tiles {
        for work in low.work(tile) {
            let TileWorkRef::Kernel(run) = work else {
                continue;
            };
            match run.kernel {
                TileKernelSpec::Cast { to, .. } if to == fp8 => {
                    casts += 1;
                    assert_eq!(
                        run.requirements.inputs[0].format.layout.order,
                        run.requirements.outputs[0].format.layout.order
                    );
                    assert_ne!(
                        run.requirements.outputs[0].format.layout.order,
                        ElementOrder::RowMajor
                    );
                    let abi = validate_kernel_run(run).unwrap();
                    assert!(scalar_values(run, &abi).unwrap()[3] > 0);
                }
                TileKernelSpec::Gemm {
                    multiply,
                    accumulate,
                    ..
                } => {
                    gemms += 1;
                    assert_eq!(multiply, fp8);
                    assert_eq!(accumulate, AccumulationPrecision::F16);
                    assert_eq!(run.requirements.outputs[0].format.precision, Precision::F16);
                    assert_eq!(
                        scalar_values(run, &validate_kernel_run(run).unwrap()).unwrap(),
                        vec![(-8i32) as u32]
                    );
                    let mut rescaled = run.clone();
                    let metadata = std::sync::Arc::make_mut(&mut rescaled.metadata);
                    if let TileKernelSpec::Gemm { multiply, .. } = &mut metadata.kernel {
                        *multiply = Precision::F8F143 { scale_exponent: 1 };
                    }
                    for input in &mut metadata.requirements.inputs {
                        input.format.precision = Precision::F8F143 { scale_exponent: 1 };
                    }
                    assert_eq!(
                        KernelSpecialization::from_run(run).unwrap(),
                        KernelSpecialization::from_run(&rescaled).unwrap()
                    );
                    assert_eq!(
                        scalar_values(&rescaled, &validate_kernel_run(&rescaled).unwrap()).unwrap(),
                        vec![2]
                    );
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
            TileKernelSpec::Gemm {
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
                mode: GemmKernelMode::Initialize,
                weights: GemmWeightLoad::Standard,
                inner_block: 64,
                output_columns: 16,
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
                outputs: vec![KernelAccess::new(format, 8)],
                distinct_elements: Vec::new(),
            },
        );
        let expected = outer
            * if transposed {
                semantic_columns
            } else {
                semantic_rows
            };
        assert_eq!(gemm_rows(&run).unwrap(), expected, "random case {case}");
    }
}

#[test]
fn randomized_gemm_abis_resolve_to_retained_symbols() {
    let mut random = fastrand::Rng::with_seed(0x6162_6921);
    for _ in 0..64 {
        let precision = if random.bool() {
            Precision::F16
        } else {
            Precision::F32
        };
        let mode = if random.bool() {
            GemmKernelMode::Initialize
        } else {
            GemmKernelMode::Accumulate
        };
        let weights = if precision == Precision::F16 && random.bool() {
            GemmWeightLoad::Interleaved
        } else {
            GemmWeightLoad::Standard
        };
        let format = TensorFormat {
            precision,
            layout: Layout {
                order: crate::ElementOrder::RowMajor,
                tiling: TensorTiling::replicated(1),
                memory_class: MemoryClass::Ipu21Standard,
            },
        };
        let operand = KernelAccess::new(format, 8);
        let requirements = KernelRequirements {
            inputs: vec![operand.clone(), operand.clone()],
            outputs: vec![operand],
            distinct_elements: Vec::new(),
        };
        let abi = tile_kernel_abi(
            &TileKernelSpec::Gemm {
                multiply: precision,
                accumulate: AccumulationPrecision::F32,
                mode,
                weights,
                inner_block: 64,
                output_columns: [32, 64, 128][random.usize(0..3)],
            },
            &requirements,
        )
        .unwrap();
        assert!(matches!(abi.symbols, KernelSymbols::Specialized));
        assert_eq!(abi.inputs, 2);
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
            crate::OperatorCandidate::Concrete(candidate) if matches!(candidate.plan.dispatch,
                crate::OperatorDispatch::BlockedGemm { orientation: crate::GemmOrientation::Normal,
                    distribution: crate::GemmDistribution::OutputStationary, .. }))
        });
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(
            &crate::expand_tiles(&mid).unwrap(),
            config.diagnostic_checkpoints,
        );
        let plan = KernelBuildPlan::from_program(&low).unwrap();
        let addresses = low
            .shards
            .iter()
            .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
            .collect::<BTreeMap<_, _>>();
        let specialization = plan
            .compilations
            .iter()
            .find(|unit| !unit.retained_symbols.is_empty())
            .unwrap();
        assert_eq!(plan.compilations.len(), 3);
        assert_eq!(
            plan.compilations
                .iter()
                .filter(|unit| unit.name.ends_with("_dispatch"))
                .count(),
            1
        );
        let planned_rows = plan
            .symbols
            .keys()
            .filter_map(|key| match key {
                KernelSpecialization::Gemm(_, _, _, _, _, rows, _) => Some(*rows),
                _ => None,
            })
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
        assert_eq!(specialization.retained_symbols.len(), 1);
        assert!(
            plan.retained_symbols()
                .all(|symbol| !symbol.contains("accumulate"))
        );
        for run in low
            .tiles
            .iter()
            .flat_map(|tile| low.work(tile))
            .filter_map(|work| {
                if let TileWorkRef::Kernel(run) = work {
                    Some(run)
                } else {
                    None
                }
            })
        {
            let call = plan.call(run).unwrap();
            assert!(plan.retained_symbols().any(|symbol| symbol == call.symbol));
            assert!(call.arguments.is_empty());
            let compute =
                materialize_kernel_run(run, &low.shards, &addresses, &plan, &BTreeMap::new())
                    .unwrap_or_else(|error| {
                        panic!("batch={batch} tiles={tiles} run={run:?}: {error}")
                    });
            assert_eq!(compute.symbol, call.symbol);
            assert_eq!(compute.input_addresses.len(), 2);
        }
    }
}

#[test]
fn randomized_gelu_abis_select_supported_layout_paths() {
    let mut random = fastrand::Rng::with_seed(0x6765_6c75);
    for _ in 0..64 {
        let tiles = 1_u16 << random.u32(0..=5);
        let input_layout = if random.bool() {
            Layout::amp_left_result(tiles)
        } else {
            Layout::row_sharded(tiles)
        };
        let output_layout = input_layout.clone();
        let requirement = |layout| {
            KernelAccess::new(
                TensorFormat {
                    precision: Precision::F16,
                    layout,
                },
                8,
            )
        };
        let requirements = KernelRequirements {
            inputs: vec![requirement(input_layout)],
            outputs: vec![requirement(output_layout)],
            distinct_elements: Vec::new(),
        };
        let abi = tile_kernel_abi(&TileKernelSpec::Gelu, &requirements).unwrap();
        assert_eq!(abi.inputs, 1);
        assert_eq!(abi.scalar_arguments, &[ScalarValue::ElementCount]);
        assert_eq!(abi.symbols, KernelSymbols::Exact("gelu_tanh_approx_f16"));
    }
}

#[test]
fn attention_stages_support_multiple_configurations_and_block_sizes() {
    let mut stages = Vec::new();
    for head in [32, 64] {
        for padded in [32, 64] {
            for rows in [1, 3, 7] {
                for keys in [1, padded / 2, padded] {
                    stages.push((
                        TileKernelSpec::AttentionSoftmax {
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
                TileKernelSpec::AttentionMerge {
                    value_dimension: values,
                    padded_value_dimension: values,
                    key_block_columns: 64,
                    initial: true,
                    final_block: false,
                },
                rows,
            ));
        }
    }
    let mut plan = KernelBuildPlan::default();
    plan.add_attention_stages(
        stages
            .iter()
            .map(|(kernel, _)| {
                KernelSpecialization::stage(
                    kernel,
                    if matches!(kernel, TileKernelSpec::AttentionSoftmax { .. }) {
                        Precision::F16
                    } else {
                        Precision::F32
                    },
                    Precision::F16,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    // Four dimension pairs each have full/masked softmax, plus two merges.
    // Query-row counts share assembly instead of producing more codelets.
    assert_eq!(plan.compilations.len(), 10);
    for (kernel, rows) in stages {
        let (inputs, expected) = match kernel {
            TileKernelSpec::AttentionSoftmax { key_columns, .. } => (1, vec![rows, key_columns, 0]),
            TileKernelSpec::AttentionMerge { .. } => (2, vec![1, 0, rows]),
            _ => unreachable!(),
        };
        let format = TensorFormat {
            precision: if inputs == 2 {
                Precision::F32
            } else {
                Precision::F16
            },
            layout: Layout::row_major(TensorTiling::replicated(1)),
        };
        let output = ShardView {
            shard: BlockValueId::from_index(0),
            extents: [rows, 16]
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
        let run = KernelRun::new(
            WorkProvenance {
                operation: None,
                value: None,
                reason: WorkReason::OperatorKernel,
            },
            kernel,
            (0..inputs).map(|_| output.clone()).collect(),
            vec![output],
            KernelRequirements {
                inputs: vec![
                    KernelAccess::new(
                        TensorFormat {
                            precision: Precision::F16,
                            ..format.clone()
                        },
                        8
                    );
                    inputs
                ],
                outputs: vec![KernelAccess::new(format, 8)],
                distinct_elements: Vec::new(),
            },
        );
        let call = plan.call(&run).unwrap();
        assert_eq!(call.arguments, expected);
        assert!(plan.retained_symbols().any(|symbol| symbol == call.symbol));
    }
}

#[test]
fn block_rearrangements_have_distinct_objects_and_symbols() {
    // These layouts have identical matrix sizes and C++ order indices, but
    // require different worker code. They must coexist in one linked package.
    let targets = [(8, 16), (16, 8), (64, 16)];
    let mut inventory = KernelInventory::default();
    for (row_block, column_block) in targets {
        inventory.rearrangements.insert((
            RearrangeTarget::BlockMajor {
                row_block,
                column_block,
            },
            128,
            128,
            128,
            128,
        ));
    }
    let plan = KernelBuildPlan::from_inventory(inventory).unwrap();
    let objects = plan
        .compilations
        .iter()
        .map(|object| &object.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(objects.len(), plan.compilations.len());
    let symbols = plan.retained_symbols().collect::<BTreeSet<_>>();
    assert_eq!(symbols.len(), targets.len());
    for (row_block, column_block) in targets {
        let symbol = &plan.symbols[&KernelSpecialization::Rearrange((
            RearrangeTarget::BlockMajor {
                row_block,
                column_block,
            },
            128,
            128,
            128,
            128,
        ))];
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
        TileKernelSpec::FillZero {
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
            outputs: vec![KernelAccess::new(tensor_type.format, 8)],
            distinct_elements: Vec::new(),
        },
    );
    let addresses = BTreeMap::from([(shard.id, 0x60000)]);
    let plan = KernelBuildPlan::default();
    let materialize = |run: &KernelRun| {
        materialize_kernel_run(
            run,
            std::slice::from_ref(&shard),
            &addresses,
            &plan,
            &BTreeMap::new(),
        )
    };
    let compute = materialize(&run).unwrap();
    assert_eq!(
        compute.output_address,
        TileAddress::Absolute(0x60000 + 32 + 16)
    );
    assert_eq!(plan.call(&run).unwrap().arguments, vec![1, 1]);
    for (offset, bytes) in [(48, 56), (1, 8), (0, 7), (u32::MAX - 7, 16)] {
        let run = KernelRun::new(
            run.provenance,
            TileKernelSpec::FillZero {
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
    let orientation = crate::GemmOrientation::Normal;
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
                crate::OperatorCandidate::Concrete(candidate) if matches!(candidate.plan.dispatch,
                    crate::OperatorDispatch::BlockedGemm { orientation: candidate_orientation,
                        distribution: crate::GemmDistribution::OutputStationary, .. } if candidate_orientation == orientation)));
            let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
            let low = lower_to_tiles(&crate::expand_tiles(&mid).unwrap(), false);
            let build = KernelBuildPlan::from_program(&low).unwrap();
            let addresses = low
                .shards
                .iter()
                .map(|shard| (shard.id, 0x60000 + shard.id.index() * 0x10000))
                .collect();
            let mut products = 0;
            for run in &low.kernel_runs {
                if !matches!(run.kernel, TileKernelSpec::Gemm { .. }) {
                    continue;
                }
                products += 1;
                assert!(gemm_rows(run).unwrap().is_multiple_of(16));
                assert_eq!(
                    run.requirements.outputs[0]
                        .format
                        .layout
                        .order
                        .gemm_output_group(),
                    Some(64)
                );
                let step =
                    materialize_kernel_run(run, &low.shards, &addresses, &build, &BTreeMap::new())
                        .unwrap();
                assert!(step.symbol.contains("packed64"));
                let source = &low.shards[run.outputs[0].shard.index() as usize];
                assert_eq!(
                    source.tensor_type.format.layout.order,
                    run.requirements.outputs[0].format.layout.order
                );
            }
            assert!(products > 0);
        }
    }
}

#[test]
fn f32_to_f16_cast_calls_cover_partial_worker_waves() {
    let plan = KernelBuildPlan::from_inventory(KernelInventory {
        exact_symbols: BTreeSet::from(["cast_f32_f16"]),
        ..KernelInventory::default()
    })
    .unwrap();
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
            TileKernelSpec::Cast {
                from: Precision::F32,
                to: Precision::F16,
            },
            vec![view(0)],
            vec![view(1)],
            KernelRequirements {
                inputs: vec![KernelAccess::new(format(Precision::F32), 8)],
                outputs: vec![KernelAccess::new(format(Precision::F16), 8)],
                distinct_elements: Vec::new(),
            },
        );
        let call = plan.call(&run).unwrap();
        assert_eq!(call.arguments, [count]);
        assert!(plan.retained_symbols().any(|symbol| symbol == call.symbol));
    }
}

#[test]
fn shared_row_tails_preserve_column_alignment_for_wide_packing() {
    // 14 columns have a two-halfword tail: a 64-bit load would cross the row.
    // Sharing kernels across row tails must not erase this distinction.
    for columns in [14, 16] {
        let shape = rearrangement_specialization(
            RearrangeTarget::BlockMajor {
                row_block: 64,
                column_block: 16,
            },
            63,
            64,
            columns,
            16,
        );
        let mut inventory = KernelInventory::default();
        inventory.rearrangements.insert(shape);
        let plan = KernelBuildPlan::from_inventory(inventory).unwrap();
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
fn worker_stack_support_follows_cpp_recipes() {
    for (symbols, expected) in [
        (vec![], false),
        (vec!["gelu_f8", "reduce_sum_f16"], false),
        (vec!["layer_norm_f16"], true),
        (
            vec!["layer_norm_f8", "add_layer_norm_moments", "add_f16"],
            true,
        ),
    ] {
        let plan = KernelBuildPlan::from_inventory(KernelInventory {
            exact_symbols: symbols.into_iter().collect(),
            ..KernelInventory::default()
        })
        .unwrap();
        assert_eq!(
            plan.compilations
                .iter()
                .filter(|unit| unit.source == "worker_support.S")
                .count(),
            usize::from(expected)
        );
    }
    // The hand-written softmax worker does not use a compiler-managed stack.
    let plan = KernelBuildPlan::from_inventory(KernelInventory {
        attention_stages: BTreeSet::from([KernelSpecialization::Softmax(
            64,
            32,
            32,
            Precision::F16,
        )]),
        ..KernelInventory::default()
    })
    .unwrap();
    assert_eq!(plan.compilations.len(), 1);
    assert_eq!(plan.compilations[0].source, "attention_softmax_f16.S");
}

#[test]
fn unsupported_kernel_abis_fail_at_lookup() {
    let format = TensorFormat {
        precision: Precision::F32,
        layout: Layout::row_sharded(1),
    };
    for kernel in [
        TileKernelSpec::Add,
        TileKernelSpec::LayerNorm,
        TileKernelSpec::Gelu,
        TileKernelSpec::Cast {
            from: Precision::F16,
            to: Precision::F32,
        },
        TileKernelSpec::Rearrange {
            from: format.layout.clone(),
            to: format.layout.clone(),
        },
    ] {
        let requirements = KernelRequirements::new(&kernel, [format.clone()], vec![format.clone()]);
        assert_eq!(
            tile_kernel_abi(&kernel, &requirements),
            Err(KernelAbiError::Unavailable(kernel))
        );
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
    let kernel = TileKernelSpec::BiasGelu;
    let mut run = KernelRun::new(
        WorkProvenance {
            operation: None,
            value: None,
            reason: WorkReason::OperatorKernel,
        },
        kernel.clone(),
        vec![view(0, [1, 2]), view(1, [1, 2])],
        vec![view(2, [1, 2])],
        KernelRequirements::new(&kernel, [format.clone(), format.clone()], vec![format]),
    );
    validate_kernel_run(&run).unwrap();
    // An unchecked u32 product wraps to the expected bias width of two.
    run.inputs[1] = view(1, [2, (1 << 31) + 1]);
    assert_eq!(
        validate_kernel_run(&run),
        Err(KernelAbiError::ElementCountOverflow)
    );
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
                TileKernelSpec::Gelu,
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
                materialize_kernel_run(
                    &run,
                    &shards,
                    &addresses,
                    &KernelBuildPlan::default(),
                    &BTreeMap::new(),
                )
                .unwrap();
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
            TileKernelSpec::Gelu,
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
        Err(KernelError::Abi(KernelAbiError::Unavailable(
            TileKernelSpec::Gelu
        )))
    ));
}
