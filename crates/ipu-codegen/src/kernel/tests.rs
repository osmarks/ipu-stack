use super::*;
use crate::{
    AccumulationPrecision, ComputeGraph, Ipu21CostModel, KernelAccess, KernelRequirements, Layout,
    MemoryClass, PipelineConfig, ShardExtent, ShardView, TensorFormat, TensorTiling,
    WorkProvenance, WorkReason, lower, lower_to_tiles,
};

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
            ShardView {
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
            },
            KernelRequirements {
                inputs: Vec::new(),
                output: KernelAccess::new(format, 8),
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
            output: operand,
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
        assert_eq!(abi.availability, KernelAvailability::Implemented);
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
        assert_eq!(plan.compilations.len(), 1);
        let planned_rows = plan
            .symbols
            .keys()
            .filter_map(|key| match key {
                KernelSpecialization::Gemm(_, _, _, _, _, rows, _) => Some(*rows),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            plan.compilations[0]
                .flags
                .iter()
                .any(|flag| flag == &format!("-DGEMM_SMALL_ROWS={}", planned_rows[0]))
        );
        assert!(
            plan.compilations[0]
                .flags
                .iter()
                .any(|flag| flag == "-DGEMM_SINGLE_ROWS=1")
        );
        assert_eq!(plan.compilations[0].retained_symbols.len(), 2);
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
            output: requirement(output_layout),
            distinct_elements: Vec::new(),
        };
        let abi = tile_kernel_abi(&TileKernelSpec::Gelu, &requirements).unwrap();
        assert_eq!(abi.availability, KernelAvailability::Implemented);
        assert_eq!(abi.inputs, 1);
        assert_eq!(abi.scalar_arguments, &[ScalarValue::ElementCount]);
        assert_eq!(
            abi.symbols,
            KernelSymbols::Exact("ipu_stack_gelu_tanh_approx_f16")
        );
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
            .map(|(kernel, _)| KernelSpecialization::stage(kernel).unwrap())
            .collect(),
    )
    .unwrap();
    // Four dimension pairs each have full/masked softmax, plus two merges.
    // Query-row counts share assembly instead of producing more codelets.
    assert_eq!(plan.compilations.len(), 10);
    for (kernel, rows) in stages {
        let (inputs, expected) = match kernel {
            TileKernelSpec::AttentionSoftmax { key_columns, .. } => (1, vec![rows, key_columns]),
            TileKernelSpec::AttentionMerge { .. } => (2, vec![1, 0, rows]),
            _ => unreachable!(),
        };
        let format = TensorFormat {
            precision: Precision::F16,
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
            (0..inputs)
                .map(|_| crate::KernelOperand {
                    views: vec![output.clone()],
                })
                .collect(),
            output,
            KernelRequirements {
                inputs: vec![KernelAccess::new(format.clone(), 8); inputs],
                output: KernelAccess::new(format, 8),
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
                object.source == "rearrange_f16.cpp"
                    && object
                        .name
                        .ends_with(&format!("_b{row_block}x{column_block}"))
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
        ShardView {
            shard: shard.id,
            extents: vec![ShardExtent {
                axis: 0,
                start: 16,
                logical_end: 64,
                physical_end: 64,
            }],
        },
        KernelRequirements {
            inputs: Vec::new(),
            output: KernelAccess::new(tensor_type.format, 8),
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
            run.output.clone(),
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
                    run.requirements
                        .output
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
                let source = &low.shards[run.output.shard.index() as usize];
                assert_eq!(
                    source.tensor_type.format.layout.order,
                    run.requirements.output.format.layout.order
                );
            }
            assert!(products > 0);
        }
    }
}
