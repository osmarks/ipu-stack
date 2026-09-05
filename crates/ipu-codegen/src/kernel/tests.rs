use super::*;
use crate::{
    AccumulationPrecision, ComputeGraph, Ipu21CostModel, Layout, MemoryClass, OperandRequirement,
    OperatorRequirements, OutputAliasing, PipelineConfig, ShardExtent, ShardView, TensorFormat,
    TensorTiling, WorkProvenance, WorkReason, lower, lower_to_tiles,
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
            TileKernel::Planned(TileKernelSpec::Gemm {
                multiply: Precision::F16,
                accumulate: AccumulationPrecision::F32,
                mode: GemmKernelMode::Initialize,
                weights: GemmWeightLoad::Standard,
                inner_block: 64,
                output_columns: 16,
            }),
            Vec::new(),
            ShardView {
                shard: LowShardId::from_index(0),
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
            KernelRequirements::Operator(OperatorRequirements {
                inputs: Vec::new(),
                output: OperandRequirement::new(format, 8),
                output_aliasing: OutputAliasing::Fresh,
                distinct_elements: Vec::new(),
            }),
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
        let operand = OperandRequirement::new(format, 8);
        let requirements = KernelRequirements::Operator(OperatorRequirements {
            inputs: vec![operand.clone(), operand.clone()],
            output: operand,
            output_aliasing: OutputAliasing::Fresh,
            distinct_elements: Vec::new(),
        });
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
        assert!(matches!(abi.symbols, KernelSymbols::GemmSpecialized));
        assert_eq!(abi.input_registers, [3, 4]);
        assert_eq!(abi.return_register, 10);
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
        let config = PipelineConfig::new(tiles)
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
        let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let low = lower_to_tiles(&mid, &config).unwrap();
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
                KernelSpecialization::Gemm(_, _, _, _, _, rows) => Some(*rows),
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
            OperandRequirement::new(
                TensorFormat {
                    precision: Precision::F16,
                    layout,
                },
                8,
            )
        };
        let requirements = KernelRequirements::Operator(OperatorRequirements {
            inputs: vec![requirement(input_layout)],
            output: requirement(output_layout),
            output_aliasing: OutputAliasing::Fresh,
            distinct_elements: Vec::new(),
        });
        let abi = tile_kernel_abi(&TileKernelSpec::Gelu, &requirements).unwrap();
        assert_eq!(abi.availability, KernelAvailability::Implemented);
        assert_eq!(abi.input_registers, [3]);
        assert_eq!(abi.scalar_arguments[0].register, 4);
        assert_eq!(
            abi.symbols,
            KernelSymbols::Exact("ipu_stack_gelu_tanh_approx_f16")
        );
    }
}
