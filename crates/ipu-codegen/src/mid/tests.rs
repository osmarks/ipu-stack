use super::planner::plan_finalists as lower_finalists;
use super::*;

const RANDOM_CASES: usize = 128;

#[test]
fn fp8_mlp_retains_quantization_before_replication() {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [1, 729, 1152]).unwrap();
    let w0 = graph.parameter("w0", [1, 1152, 4304]).unwrap();
    let w1 = graph.parameter("w1", [1, 4304, 1152]).unwrap();
    let h = graph.gemm(x, w0).unwrap();
    let h = graph.gelu(h).unwrap();
    let y = graph.gemm(h, w1).unwrap();
    graph.set_outputs([y]).unwrap();
    let fp8 = Precision::F8F143 { scale_exponent: -4 };
    let mut config = PipelineConfig::new(1472)
        .with_automatic_input(x, fp8)
        .with_automatic_input(w0, fp8)
        .with_automatic_input(w1, fp8);
    config
        .operator_candidates
        .retain(|c| !matches!(c.operator(), MidOperator::Gemm { .. }));
    config
        .operator_candidates
        .push(OperatorCandidate::fp8_gemm(1472, -4));
    let finalists = lower_finalists(&graph, &config, &crate::Ipu21CostModel, 4).unwrap();
    assert!(finalists.iter().any(|mid| {
        mid.operations
            .iter()
            .filter_map(|op| op.conversion_plan())
            .any(|plan| {
                plan.input.format.precision == Precision::F16
                    && plan.output.format.precision == fp8
                    && plan.output.format.layout.tiling.replicas == 1
            })
    }));
}

#[test]
fn fp8_conversion_precedes_operand_replication() {
    let mut graph = ComputeGraph::new();
    let host = graph.host_input("input", [512, 64]).unwrap();
    graph.gelu(host).unwrap();
    let mut state = planner::LoweringState::default();
    let input_layout = Layout::amp_left(64, 64);
    let input = state.value(
        ValueId::from_index(0),
        TensorType::new([512, 64], Precision::F16, input_layout.clone()),
    );
    let target = TensorFormat {
        precision: Precision::F8F143 { scale_exponent: -4 },
        layout: Layout {
            tiling: TensorTiling::replicated(64),
            ..input_layout.clone()
        },
    };
    let mut operations = Vec::new();
    planner::ensure_format(
        input,
        target.clone(),
        OperandMaterialization::Complete,
        graph.operations()[0].id,
        &crate::Ipu21CostModel,
        &mut state,
        &mut operations,
    );
    assert_eq!(operations.len(), 2);
    let cast = operations[0].conversion_plan().unwrap();
    assert_eq!(cast.input.format.layout, input_layout);
    assert_eq!(cast.output.format.layout, input_layout);
    assert_eq!(cast.output.format.precision, target.precision);
    let exchange = operations[1].conversion_plan().unwrap();
    assert_eq!(exchange.input.format.precision, target.precision);
    assert_eq!(exchange.output.format, target);
}

#[test]
fn randomized_memory_peaks_allow_noncoincident_classes() {
    let mut random = fastrand::Rng::with_seed(0x636c_6173_735f_7372);
    let capacity = u64::from(crate::memory::IPU21_PLANNED_DATA_BYTES);
    let interleaved_capacity = u64::from(crate::memory::IPU21_INTERLEAVED_REGION_BYTES);
    let element = u64::from(ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE);
    let mut admitted_noncoincident_peaks = 0;
    for _ in 0..RANDOM_CASES * 16 {
        let standard = random.u64(0..=capacity);
        let interleaved = random.u64(0..=interleaved_capacity);
        let reservation = random.u64(0..=capacity / 4);
        let simultaneous =
            random.u64(standard.max(interleaved)..=standard.saturating_add(interleaved));
        let peaks = MemoryPeaks {
            standard,
            interleaved,
            total: simultaneous,
            maximum_standard_allocation: 0,
            ..MemoryPeaks::default()
        };
        let aligned_interleaved = interleaved.div_ceil(element) * element;
        let static_partition = standard
            .saturating_add(aligned_interleaved)
            .saturating_add(reservation);
        let fits = peaks.fits_ipu21_with_budget(reservation, capacity);
        assert_eq!(fits, simultaneous.saturating_add(reservation) <= capacity);
        if simultaneous.saturating_add(reservation) <= capacity && static_partition > capacity {
            admitted_noncoincident_peaks += 1;
            assert!(fits);
        }
    }
    assert!(admitted_noncoincident_peaks > 0);
}

fn dimension(random: &mut fastrand::Rng) -> u32 {
    random.u32(1..=128)
}

fn small_dimension(random: &mut fastrand::Rng) -> u32 {
    random.u32(1..=4)
}

fn precision(random: &mut fastrand::Rng) -> Precision {
    if random.bool() {
        Precision::F16
    } else {
        Precision::F32
    }
}

fn format(precision: Precision, layout: Layout) -> TensorFormat {
    TensorFormat { precision, layout }
}

fn random_format(random: &mut fastrand::Rng, tiles: u16) -> TensorFormat {
    let tiling = if random.bool() {
        TensorTiling::replicated(tiles)
    } else {
        TensorTiling::sharded(TensorAxis::FromEnd(2), tiles)
    };
    let mut layout = Layout::row_major(tiling);
    if random.bool() {
        layout.memory_class = MemoryClass::Ipu21Interleaved;
    }
    format(precision(random), layout)
}

#[test]
fn randomized_future_state_is_id_independent_but_preserves_aliasing() {
    let mut random = fastrand::Rng::with_seed(0x616c_6961_7365_7321);
    for _ in 0..RANDOM_CASES {
        let mut graph = ComputeGraph::new();
        let first = graph.host_input("first", [1]).unwrap();
        let second = graph.host_input("second", [1]).unwrap();
        let dummy = graph.host_input("dummy", [1]).unwrap();
        let tiles = random.u16(1..=64);
        let tensor_type = TensorType {
            shape: TensorShape::new([random.u32(1..=128)]),
            format: random_format(&mut random, tiles),
        };
        let aliases = random.bool();
        let automatic = random.bool();
        let parameter = random.bool();

        let make_branch = |prepend_dummy: bool, aliases: bool| {
            let mut state = LoweringState::default();
            if prepend_dummy {
                state.value(dummy, tensor_type.clone());
            }
            let first_id = state.value(first, tensor_type.clone());
            let second_id = if aliases {
                state.value_in_storage_group(second, tensor_type.clone(), first_id)
            } else {
                state.value(second, tensor_type.clone())
            };
            if automatic {
                state.automatic_inputs.extend([first_id, second_id]);
            }
            if parameter {
                state.parameter_values.extend([first_id, second_id]);
            }
            BeamBranch {
                values: [(first, first_id), (second, second_id)]
                    .into_iter()
                    .collect(),
                state,
                operations: Vec::new(),
                peak_memory: MemoryPeaks::default(),
                analysis: std::sync::OnceLock::new(),
            }
        };
        let future = [first, second].into_iter().collect();
        let constraints = RegionPlanningConstraints {
            allocation_copies: [(first, random.u32(1..=8))].into_iter().collect(),
            required_equal_formats: vec![(first, second)],
        };
        let baseline = future_beam_state(&make_branch(false, aliases), &future, &constraints);
        let renumbered = future_beam_state(&make_branch(true, aliases), &future, &constraints);
        let changed_aliasing =
            future_beam_state(&make_branch(true, !aliases), &future, &constraints);
        assert_eq!(baseline, renumbered);
        assert_ne!(baseline, changed_aliasing);
    }
}

#[test]
fn randomized_active_tile_candidates_bound_idle_capacity() {
    let mut random = fastrand::Rng::with_seed(0x7469_6c65);
    for _ in 0..RANDOM_CASES {
        let capacity = random.u16(1..=1472);
        let counts = candidate_active_tile_counts(capacity);
        assert_eq!(counts[0], capacity);
        assert!(counts.windows(2).all(|pair| pair[0] > pair[1]));
        assert!(counts.iter().all(|&count| count <= capacity));
        assert!(counts[1..].iter().all(|count| count.is_power_of_two()));
        assert_eq!(counts.last(), Some(&1));
    }
    for exponent in 1..=10 {
        let capacity = 1_u16 << exponent;
        assert_eq!(candidate_active_tile_counts(capacity).len(), exponent + 1);
    }
}

#[test]
fn randomized_shape_aware_tile_candidates_follow_graph_extents() {
    let mut random = fastrand::Rng::with_seed(0x7368_6170_655f_6772);
    for case in 0..RANDOM_CASES {
        let capacity = random.u16(16..=1472);
        let extent = random.u16(2..=capacity);
        let shape = TensorShape(vec![u32::from(extent), random.u32(1..=4096)]);
        let counts = shape_aware_active_tile_counts(capacity, [&shape]);
        let expected = capacity / extent * extent;
        if expected >= capacity.div_ceil(2) && expected < capacity {
            assert!(counts.contains(&expected), "case {case}");
        }
        assert!(counts.iter().all(|&count| {
            count < capacity
                && count >= capacity.div_ceil(2)
                && shape.0.iter().any(|&axis| u32::from(count) % axis == 0)
        }));
    }
}

fn value(lowered: &MidProgram, id: MidValueId) -> &MidValue {
    &lowered.values[id.index() as usize]
}

#[test]
fn randomized_parallel_reduction_candidates_cover_uneven_three_axis_grids() {
    let mut random = fastrand::Rng::with_seed(0x7061_7274_6961_6c73);
    let mut distributed_result_cases = 0;
    for _ in 0..RANDOM_CASES {
        let output_columns = AMP_OUTPUT_COLUMN_BLOCK;
        let inner_partitions = random.u16(2..=4);
        let column_partitions = random.u16(1..=4);
        let row_partitions = random.u16(1..=8);
        let tiles = row_partitions * column_partitions * inner_partitions;
        let k = u32::from(inner_partitions) * 64 + random.u32(0..64);
        let n = u32::from(column_partitions) * output_columns + random.u32(0..output_columns);
        let m = u32::from(row_partitions) + random.u32(0..=16);
        let inputs = [
            TensorType::new([m, k], Precision::F16, Layout::row_sharded(tiles)),
            TensorType::new([k, n], Precision::F16, Layout::row_sharded(tiles)),
        ];
        let config = PipelineConfig::new(tiles).with_planning_beam_width(16);
        let candidates = parallel_reduction_candidates(
            OperatorCandidate::parallel_gemm(tiles).operator(),
            tiles,
            &inputs,
            &TensorShape(vec![m, n]),
            &config,
            &Ipu21CostModel,
            true,
            None,
            None,
            &[],
        );
        assert!(
            !candidates.is_empty(),
            "shape={m}x{k}x{n} tiles={tiles} output_columns={output_columns}"
        );
        distributed_result_cases += usize::from(candidates.iter().any(|candidate| {
            matches!(
                candidate.dispatch,
                OperatorDispatch::BlockedGemm {
                    distribution: GemmDistribution::ParallelReduction {
                        result_row_partitions,
                        result_column_partitions,
                        ..
                    },
                    ..
                } if (result_row_partitions, result_column_partitions) != (1, 1)
            )
        }));
        for candidate in candidates {
            assert!(
                candidate.supports(&inputs, &TensorShape(vec![m, n])),
                "unsupported candidate: {candidate:?}; shape={m}x{k}x{n}"
            );
            assert!(matches!(
                candidate.dispatch,
                OperatorDispatch::BlockedGemm {
                    inner_block,
                    output_column_block,
                    orientation,
                    distribution: GemmDistribution::ParallelReduction {
                        row_partitions: actual_rows,
                        column_partitions: actual_columns,
                        inner_partitions: actual,
                        ..
                    },
                    ..
                } if actual_rows * actual_columns * actual <= tiles
                    && actual_rows * actual_columns * actual >= tiles.div_ceil(2)
                    && u32::from(actual_rows) <= match orientation {
                        GemmOrientation::Normal => m,
                        GemmOrientation::Swapped => n,
                    }
                    && u32::from(actual_columns) * output_column_block >= match orientation {
                        GemmOrientation::Normal => n,
                        GemmOrientation::Swapped => m,
                    }
                    && u32::from(actual) * inner_block >= k
            ));
        }
    }
    assert!(distributed_result_cases > 0);
}

#[test]
fn randomized_cycle_model_rewards_direct_interleaved_weight_loads() {
    let mut random = fastrand::Rng::with_seed(0x6379_636c);
    for _ in 0..RANDOM_CASES {
        let rows = 1_u16 << random.u32(0..=2);
        let columns = 1_u16 << random.u32(0..=2);
        let tiles = rows * columns;
        let m = u32::from(rows) * random.u32(1..=4);
        let k = 64 * random.u32(2..=4);
        let n = u32::from(columns) * 64;
        let left = TensorType::new(
            [m, k],
            Precision::F16,
            Layout::amp_left_grid(64, tiles, rows, columns, GridOrder::ColumnsFast),
        );
        let mut standard_layout =
            Layout::block_major_matrix_grid(64, 64, tiles, rows, columns, GridOrder::ColumnsFast);
        let mut direct_layout = standard_layout.clone();
        direct_layout.memory_class = MemoryClass::Ipu21Interleaved;
        standard_layout.memory_class = MemoryClass::Ipu21Standard;
        let standard = TensorType::new([k, n], Precision::F16, standard_layout);
        let direct = TensorType::new([k, n], Precision::F16, direct_layout);
        let output = TensorType::new(
            [m, n],
            Precision::F16,
            Layout::amp_output_grid(64, tiles, rows, columns, GridOrder::ColumnsFast),
        );
        let operator = MidOperator::Gemm {
            options: GemmOptions::default(),
            multiply: Precision::F16,
            accumulate: AccumulationPrecision::F32,
        };
        let dispatch = default_dispatch(operator);
        let requirements = StorageRequirements {
            inputs: vec![
                OperandRequirement::new(left.format.clone(), 32),
                OperandRequirement::new(standard.format.clone(), 32),
            ],
            output: OperandRequirement::new(output.format.clone(), 8),
            output_aliasing: OutputAliasing::Fresh,
            distinct_elements: Vec::new(),
        };
        let plan = OperatorPlan {
            operator,
            dispatch,
            requirements,
            deferred_output: None,
        };
        let standard_cost =
            Ipu21CostModel.operator_cycles(&plan, &[left.clone(), standard], &output);
        let direct_cost = Ipu21CostModel.operator_cycles(&plan, &[left, direct], &output);
        assert!(direct_cost < standard_cost);
    }
}

#[test]
fn randomized_parameter_storage_balances_one_copy_independently_of_compute_grids() {
    let mut random = fastrand::Rng::with_seed(0x6f77_6e65_7273);
    for case in 0..RANDOM_CASES {
        let row_partitions = 1_u16 << random.u32(1..=4);
        let column_partitions = 1_u16 << random.u32(0..=4);
        let tiles = row_partitions * column_partitions;
        let inner_blocks = u32::from(row_partitions) * random.u32(1..=4);
        let inner = inner_blocks * AMP_INNER_BLOCK;
        let columns = u32::from(column_partitions) * AMP_OUTPUT_COLUMN_BLOCK;
        let grid = AmpGridShape {
            tile_count: tiles,
            row_partitions,
            column_partitions,
            order: GridOrder::ColumnsFast,
        };
        let candidate = amp_grid_gemm_operator_candidate(
            Precision::F16,
            64,
            16,
            AMP_OUTPUT_COLUMN_BLOCK,
            grid,
            AmpWeightPlacement::resident(MemoryClass::Ipu21Interleaved),
        );
        let inputs = [
            TensorType::new(
                [u32::from(row_partitions), inner],
                Precision::F16,
                candidate.plan.requirements.inputs[0].format.layout.clone(),
            ),
            TensorType::new(
                [inner, columns],
                Precision::F16,
                candidate.plan.requirements.inputs[1].format.layout.clone(),
            ),
        ];
        let variants =
            independent_parameter_storage(&candidate.plan, &inputs, 1, &PipelineConfig::new(tiles));
        assert!(!variants.is_empty(), "case {case}");
        for variant in variants {
            let tiling = &variant.requirements.inputs[1].format.layout.tiling;
            assert_eq!(tiling.replicas, 1, "case {case}");
            assert!(tiling.tile_count <= tiles, "case {case}");
            assert_eq!(
                tiling.tile_count,
                tiling.replicas
                    * tiling
                        .axes
                        .iter()
                        .map(|axis| axis.partitions)
                        .product::<u16>(),
                "case {case}"
            );
            assert!(
                variant.requirements.inputs[1]
                    .format
                    .layout
                    .padded_shape(&inputs[1].shape)
                    .is_ok(),
                "case {case}"
            );
        }
    }
}

fn assert_conversions_are_explicit(lowered: &MidProgram, operations: &[MidOperation]) {
    for operation in operations {
        let [input] = operation.inputs.as_slice() else {
            continue;
        };
        let [result] = operation.results.as_slice() else {
            continue;
        };
        let before = &value(lowered, *input).tensor_type;
        let after = &value(lowered, *result).tensor_type;
        if let MidOperationKind::Convert(plan) = &operation.kind {
            assert_eq!(plan.input.format, before.format);
            assert_eq!(plan.output.format, after.format);
            assert_eq!(before.shape, after.shape);
            if before.format.precision != after.format.precision {
                assert_eq!(before.format.layout, after.format.layout);
                assert_eq!(plan.strategy, ConversionStrategy::LocalKernel);
            } else {
                assert_ne!(before.format.layout, after.format.layout);
            }
        }
    }
}

fn assert_operator_signature(
    lowered: &MidProgram,
    operation: &MidOperation,
    inputs: &[TensorFormat],
    output: TensorFormat,
) {
    assert_eq!(operation.inputs.len(), inputs.len());
    for (&value_id, expected) in operation.inputs.iter().zip(inputs) {
        assert_eq!(&value(lowered, value_id).tensor_type.format, expected);
    }
    assert_eq!(
        value(lowered, operation.results[0]).tensor_type.format,
        output
    );
}

struct ColumnParityCost;

impl CostModel for ColumnParityCost {
    fn operator_cycle_override(
        &self,
        plan: &OperatorPlan,
        _inputs: &[TensorType],
        output: &TensorType,
    ) -> Option<u64> {
        let preferred = if output.shape.0.last().unwrap().is_multiple_of(2) {
            Precision::F16
        } else {
            Precision::F32
        };
        Some(match plan.operator {
            MidOperator::Gemm { multiply, .. } if multiply == preferred => 0,
            MidOperator::Gemm { .. } => 1,
            _ => 0,
        })
    }

    fn cast_cycles(&self, _input: &TensorType, _to: Precision) -> u64 {
        0
    }

    fn rearrangement_cost(
        &self,
        _shape: &TensorShape,
        _precision: Precision,
        _strategy: ConversionStrategy,
        _from: &Layout,
        _to: &Layout,
    ) -> crate::estimate::RearrangementCost {
        crate::estimate::RearrangementCost::default()
    }
}

#[test]
fn randomized_axis_tiling_applies_or_rejects_padding() {
    let mut random = fastrand::Rng::with_seed(0x7469_6c65);
    for case in 0..RANDOM_CASES {
        let rank = random.usize(1..=6);
        let axis = random.usize(0..rank);
        let extent = dimension(&mut random);
        let block_size = random.u32(1..=32);
        let partitions = random.u16(1..=16);
        let replicas = random.u16(1..=4);
        let padding = if random.bool() {
            Padding::Reject
        } else {
            Padding::Zero
        };
        let mut shape = (0..rank)
            .map(|_| dimension(&mut random))
            .collect::<Vec<_>>();
        shape[axis] = extent;
        let layout = Layout::row_major(TensorTiling {
            tile_count: partitions * replicas,
            replicas,
            axes: vec![AxisTiling::new(
                TensorAxis::FromStart(axis as u16),
                partitions,
                block_size,
                padding,
            )],
        });

        let result = layout.padded_shape(&TensorShape(shape.clone()));
        if padding == Padding::Reject && !extent.is_multiple_of(block_size) {
            assert!(
                matches!(result, Err(LayoutError::IndivisibleAxis { .. })),
                "random case {case}"
            );
        } else {
            let padded = result.unwrap();
            let expected = extent.div_ceil(block_size) * block_size;
            assert_eq!(padded.0[axis], expected, "random case {case}");
            for (other, original) in shape.iter().enumerate() {
                if other != axis {
                    assert_eq!(padded.0[other], *original, "random case {case}");
                }
            }
        }
    }
}

#[test]
fn randomized_gemm_lowering_makes_every_format_boundary_explicit() {
    let mut random = fastrand::Rng::with_seed(0x6d69_6467);
    for case in 0..RANDOM_CASES {
        let tiles = random.u16(1..=64);
        let (rows, inner, columns) = (
            u32::from(tiles) * small_dimension(&mut random),
            random.u32(1..=2) * 64,
            small_dimension(&mut random) * 64,
        );
        let batches = (0..random.usize(0..=3))
            .map(|_| random.u32(1..=2))
            .collect::<Vec<_>>();
        let multiply = precision(&mut random);
        let left_format = format(multiply, Layout::amp_left(64, tiles));
        let right_format = format(
            multiply,
            Layout::block_major_matrix_storage(
                64,
                AMP_OUTPUT_COLUMN_BLOCK,
                tiles,
                1,
                1,
                MemoryClass::Ipu21Standard,
            ),
        );
        let output_format = format(
            multiply,
            if multiply == Precision::F16 {
                Layout::amp_left_result(tiles)
            } else {
                Layout::amp_output(tiles)
            },
        );
        let accumulate = gemm_accumulation_precision(multiply);
        let candidate = ConcreteOperatorCandidate::new(
            MidOperator::Gemm {
                options: GemmOptions::default(),
                multiply,
                accumulate,
            },
            [
                OperandRequirement::new(left_format.clone(), 32),
                OperandRequirement::new(right_format.clone(), 32),
            ],
            OperandRequirement::new(output_format.clone(), 32),
        );
        let mut left_shape = batches.clone();
        left_shape.extend([rows, inner]);
        let mut right_shape = vec![1; batches.len()];
        right_shape.extend([inner, columns]);

        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", left_shape).unwrap();
        let right = graph.parameter("right", right_shape).unwrap();
        let product = graph.gemm(left, right).unwrap();
        graph.set_outputs([product]).unwrap();
        let linear = Layout::row_sharded(tiles);
        let mut config = PipelineConfig::new(tiles)
            .with_input(left, format(precision(&mut random), linear.clone()))
            .with_input(right, format(precision(&mut random), linear));
        config.operator_candidates = vec![candidate.clone()]
            .into_iter()
            .map(OperatorCandidate::Concrete)
            .collect();

        let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap_or_else(|error| {
            panic!(
                "random case {case}: tiles={tiles} rows={rows} inner={inner} columns={columns} batches={batches:?}: {error:?}"
            )
        });
        let operator = lowered
            .operations
            .iter()
            .find(|operation| matches!(operation.kind, MidOperationKind::Operator { .. }))
            .unwrap();
        let MidOperationKind::Operator {
            plan:
                OperatorPlan {
                    operator:
                        MidOperator::Gemm {
                            multiply: selected_multiply,
                            accumulate: selected_accumulate,
                            ..
                        },
                    ..
                },
            ..
        } = operator.kind
        else {
            panic!("random case {case}: expected GEMM");
        };
        assert_eq!(selected_multiply, multiply, "random case {case}");
        assert_eq!(selected_accumulate, accumulate, "random case {case}");
        assert_eq!(
            &value(&lowered, operator.inputs[0]).tensor_type.format,
            &candidate.plan.requirements.inputs[0].format,
            "random case {case}"
        );
        let selected_right = &value(&lowered, operator.inputs[1]).tensor_type.format;
        assert_eq!(
            selected_right.precision, candidate.plan.requirements.inputs[1].format.precision,
            "random case {case}"
        );
        assert_eq!(
            selected_right.layout.order, candidate.plan.requirements.inputs[1].format.layout.order,
            "random case {case}"
        );
        let output = value(&lowered, lowered.outputs[0]);
        let expected_shape = graph.value_shape(product).unwrap().clone();
        assert_eq!(
            output.tensor_type.shape, expected_shape,
            "random case {case}"
        );
        assert_eq!(
            &output.tensor_type.format, &candidate.plan.requirements.output.format,
            "random case {case}"
        );
        assert_conversions_are_explicit(&lowered, &lowered.operations);
    }
}

#[test]
fn randomized_beam_search_preserves_formats_needed_by_later_operators() {
    let mut random = fastrand::Rng::with_seed(0x6265_616d);
    for case in 0..RANDOM_CASES {
        let tiles = [1, 2, 4, 8][random.usize(0..4)];
        let rows = u32::from(tiles) * random.u32(1..=8);
        let inner = random.u32(1..=4) * 64;
        let columns = random.u32(1..=4) * 64;
        let row = format(Precision::F16, Layout::row_sharded(tiles));
        let left = format(Precision::F16, Layout::amp_left(64, tiles));
        let right = format(
            Precision::F16,
            Layout::block_major_matrix_storage(
                64,
                AMP_OUTPUT_COLUMN_BLOCK,
                tiles,
                1,
                1,
                MemoryClass::Ipu21Standard,
            ),
        );
        let output = format(Precision::F16, Layout::amp_left_result(tiles));

        let mut graph = ComputeGraph::new();
        let activation = graph.host_input("activation", [rows, inner]).unwrap();
        let weights = graph.parameter("weights", [inner, columns]).unwrap();
        let activated = graph.gelu(activation).unwrap();
        let product = graph.gemm(activated, weights).unwrap();
        graph.set_outputs([product]).unwrap();

        let candidates = [
            ConcreteOperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(row.clone(), 8)],
                OperandRequirement::new(row.clone(), 8),
            ),
            ConcreteOperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(row.clone(), 8)],
                OperandRequirement::new(left.clone(), 8),
            ),
            ConcreteOperatorCandidate::new(
                MidOperator::Gemm {
                    options: GemmOptions::default(),
                    multiply: Precision::F16,
                    accumulate: AccumulationPrecision::F32,
                },
                [
                    OperandRequirement::new(left.clone(), 32),
                    OperandRequirement::new(right.clone(), 32),
                ],
                OperandRequirement::new(output, 32),
            ),
        ];
        let make_config = |beam_width| {
            let mut config = PipelineConfig::new(tiles)
                .with_input(activation, row.clone())
                .with_input(weights, right.clone())
                .with_planning_beam_width(beam_width);
            config.operator_candidates = candidates
                .iter()
                .cloned()
                .map(OperatorCandidate::Concrete)
                .collect();
            config
        };
        let greedy = lower(&graph, &make_config(1), &Ipu21CostModel).unwrap();
        let searched_config = make_config(2);
        let finalists = lower_finalists(&graph, &searched_config, &Ipu21CostModel, 2).unwrap();
        assert!(
            !finalists.is_empty() && finalists.len() <= 2,
            "random case {case}"
        );
        for finalist in &finalists {
            let program = expand_tiles(finalist).unwrap();
            let cycles = crate::estimate::program_cycles(&program, None).unwrap();
            assert_eq!(program.estimated_cycles, cycles.total, "random case {case}");
            assert_eq!(
                program.estimated_exchange_cycles, cycles.exchange,
                "random case {case}"
            );
            assert!(
                finalist.estimated_exchange_cycles <= finalist.estimated_cycles,
                "random case {case}"
            );
        }
        let searched = &finalists[0];

        assert!(
            searched.estimated_cycles < greedy.estimated_cycles,
            "random case {case}"
        );
        let gelu = searched
            .operations
            .iter()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::Gelu,
                            ..
                        },
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            value(searched, gelu.results[0]).tensor_type.format,
            left,
            "random case {case}"
        );
        assert!(
            searched.peak_memory.fits_ipu21_with_budget(
                searched_config.standard_memory_reservation_bytes,
                searched_config.tile_memory_budget_bytes,
            ),
            "random case {case}"
        );
    }
}

#[test]
fn randomized_gemm_lowering_rejects_per_batch_weights() {
    let mut random = fastrand::Rng::with_seed(0x6261_7463);
    for _ in 0..RANDOM_CASES {
        let batch = random.u32(2..=8);
        let rows = random.u32(1..=8);
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [batch, rows, 64]).unwrap();
        let right = graph.parameter("right", [batch, 64, 64]).unwrap();
        let output = graph.gemm(left, right).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(1)
            .with_automatic_input(left, Precision::F16)
            .with_automatic_input(right, Precision::F16);
        assert!(matches!(
            lower(&graph, &config, &Ipu21CostModel),
            Err(LoweringError::UnsupportedGemmBatching(_))
        ));
    }
}

#[test]
fn randomized_gemms_choose_precision_independently_within_one_graph() {
    let mut random = fastrand::Rng::with_seed(0x6d75_6c74);
    for case in 0..RANDOM_CASES / 4 {
        let tiles = random.u16(1..=64);
        let rows = u32::from(tiles) * small_dimension(&mut random);
        let inner = random.u32(1..=64);
        let even_columns = random.u32(1..=16) * 2;
        let odd_columns = random.u32(1..=16) * 2 - 1;
        let layout = Layout::row_sharded(tiles);
        let mut graph = ComputeGraph::new();
        let left = graph.host_input("left", [rows, inner]).unwrap();
        let even_right = graph.parameter("even", [inner, even_columns]).unwrap();
        let odd_right = graph.parameter("odd", [inner, odd_columns]).unwrap();
        let even = graph.gemm(left, even_right).unwrap();
        let odd = graph.gemm(left, odd_right).unwrap();
        graph.set_outputs([even, odd]).unwrap();
        let input_format = format(precision(&mut random), layout);
        let config = PipelineConfig::new(tiles)
            .with_input(left, input_format.clone())
            .with_input(even_right, input_format.clone())
            .with_input(odd_right, input_format);

        let lowered = lower(&graph, &config, &ColumnParityCost).unwrap();
        let chosen = lowered
            .operations
            .iter()
            .filter_map(|operation| match operation.kind {
                MidOperationKind::Operator {
                    plan:
                        OperatorPlan {
                            operator: MidOperator::Gemm { multiply, .. },
                            ..
                        },
                    ..
                } => Some(multiply),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            chosen,
            vec![Precision::F16, Precision::F32],
            "random case {case}"
        );
        for operation in lowered.operations.iter().filter(|operation| {
            matches!(
                operation.kind,
                MidOperationKind::Operator {
                    plan: OperatorPlan {
                        operator: MidOperator::Gemm { .. },
                        ..
                    },
                    ..
                }
            )
        }) {
            let requirements = &operation.operator_plan().unwrap().requirements;
            assert!(
                requirements
                    .inputs
                    .iter()
                    .chain([&requirements.output])
                    .all(|requirement| requirement.alignment == 32)
            );
            assert_eq!(
                requirements.output.format.layout.memory_class,
                MemoryClass::Ipu21Interleaved
            );
            let orientation = match operation.operator_plan().map(|plan| &plan.dispatch) {
                Some(OperatorDispatch::BlockedGemm { orientation, .. }) => *orientation,
                _ => unreachable!(),
            };
            let physical_left = match orientation {
                GemmOrientation::Normal => 0usize,
                GemmOrientation::Swapped => 1usize,
            };
            assert_eq!(
                requirements.distinct_elements,
                [vec![
                    MemoryOperand::Output,
                    MemoryOperand::Input(physical_left as u16),
                ]]
            );
            let expected_tail = match operation.kind {
                MidOperationKind::Operator {
                    plan:
                        OperatorPlan {
                            operator:
                                MidOperator::Gemm {
                                    multiply: Precision::F16,
                                    ..
                                },
                            ..
                        },
                    ..
                } => 16,
                MidOperationKind::Operator {
                    plan:
                        OperatorPlan {
                            operator:
                                MidOperator::Gemm {
                                    multiply: Precision::F32,
                                    ..
                                },
                            ..
                        },
                    ..
                } => 32,
                _ => unreachable!(),
            };
            assert_eq!(requirements.inputs[0].access_tail_bytes, expected_tail);
        }
    }
}

#[test]
fn randomized_non_gemm_lowering_honors_operator_plans() {
    let mut random = fastrand::Rng::with_seed(0x6164_642b);
    for case in 0..RANDOM_CASES {
        let tiles = random.u16(1..=64);
        let batch = random.u32(1..=2);
        let query_rows = u32::from(tiles) * random.u32(1..=2);
        let key_rows = random.u32(1..=8);
        let channels = random.u32(1..=8);
        let value_channels = random.u32(1..=8);
        let mut graph = ComputeGraph::new();
        let activation = graph
            .host_input("activation", [batch, query_rows, channels])
            .unwrap();
        let residual = graph
            .host_input("residual", [batch, query_rows, channels])
            .unwrap();
        let query = graph
            .host_input("query", [batch, query_rows, channels])
            .unwrap();
        let key = graph
            .host_input("key", [batch, key_rows, channels])
            .unwrap();
        let attention_value = graph
            .host_input("value", [batch, key_rows, value_channels])
            .unwrap();
        let activated = graph.gelu(activation).unwrap();
        let sum = graph.add(activated, residual).unwrap();
        let attended = graph.flash_attention(query, key, attention_value).unwrap();
        graph.set_outputs([sum, attended]).unwrap();

        let gelu_input = random_format(&mut random, tiles);
        let gelu_output = gelu_input.clone();
        let add_left = random_format(&mut random, tiles);
        let add_right = add_left.clone();
        let add_output = add_left.clone();
        let attention_query = format(
            Precision::F16,
            Layout::row_major(TensorTiling::replicated(tiles)),
        );
        let attention_key = attention_query.clone();
        let attention_value_format = attention_query.clone();
        let attention_output = format(Precision::F32, attention_query.layout.clone());
        let attention_accumulate = AccumulationPrecision::F32;
        let mut config = PipelineConfig::new(tiles)
            .with_input(activation, random_format(&mut random, tiles))
            .with_input(residual, random_format(&mut random, tiles))
            .with_input(query, random_format(&mut random, tiles))
            .with_input(key, random_format(&mut random, tiles))
            .with_input(attention_value, random_format(&mut random, tiles));
        config.operator_candidates = vec![
            ConcreteOperatorCandidate::new(
                MidOperator::Gelu,
                [OperandRequirement::new(gelu_input.clone(), 8)],
                OperandRequirement::new(gelu_output.clone(), 8),
            )
            .with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0])),
            ConcreteOperatorCandidate::new(
                MidOperator::Add,
                [
                    OperandRequirement::new(add_left.clone(), 8),
                    OperandRequirement::new(add_right.clone(), 8),
                ],
                OperandRequirement::new(add_output.clone(), 8),
            )
            .with_output_aliasing(OutputAliasing::MayAliasInputs(vec![0])),
            ConcreteOperatorCandidate::new(
                MidOperator::FlashAttention {
                    options: AttentionOptions::default(),
                    accumulate: attention_accumulate,
                },
                [
                    OperandRequirement::new(attention_query.clone(), 8),
                    OperandRequirement::new(attention_key.clone(), 8),
                    OperandRequirement::new(attention_value_format.clone(), 8),
                ],
                OperandRequirement::new(attention_output.clone(), 8),
            ),
        ]
        .into_iter()
        .map(OperatorCandidate::Concrete)
        .collect();

        let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let operators = lowered
            .operations
            .iter()
            .filter(|operation| matches!(operation.kind, MidOperationKind::Operator { .. }))
            .collect::<Vec<_>>();
        assert_eq!(operators.len(), 3, "random case {case}");
        let gelu = operators
            .iter()
            .copied()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::Gelu,
                            ..
                        },
                        ..
                    }
                )
            })
            .expect("random graph retains its GeLU");
        let add = operators
            .iter()
            .copied()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::Add,
                            ..
                        },
                        ..
                    }
                )
            })
            .expect("random graph retains its add");
        let attention = operators
            .iter()
            .copied()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::FlashAttention { .. },
                            ..
                        },
                        ..
                    }
                )
            })
            .expect("random graph retains its attention");
        assert_operator_signature(&lowered, gelu, &[gelu_input], gelu_output.clone());
        assert_eq!(
            gelu.operator_plan().unwrap().requirements.output_aliasing,
            OutputAliasing::MayAliasInputs(vec![0])
        );
        assert_operator_signature(&lowered, add, &[add_left, add_right], add_output);
        assert_eq!(
            add.operator_plan().unwrap().requirements.output_aliasing,
            OutputAliasing::MayAliasInputs(vec![0])
        );
        assert!(matches!(
            attention.kind,
            MidOperationKind::Operator { plan: OperatorPlan { operator: MidOperator::FlashAttention { options, .. }, .. }, .. }
                if options == AttentionOptions::default()
        ));
        assert_eq!(
            value(&lowered, attention.results[0]).tensor_type.shape.0,
            vec![batch, query_rows, value_channels],
            "random case {case}"
        );
        assert_conversions_are_explicit(&lowered, &lowered.operations);
    }
}

#[test]
fn randomized_repeat_lowering_retains_sequences_without_unrolling() {
    let mut random = fastrand::Rng::with_seed(0x7265_7065);
    for case in 0..RANDOM_CASES {
        let tiles = random.u16(1..=64);
        let size = u32::from(tiles);
        let count = random.u32(1..=12);
        let layout = Layout::row_sharded(tiles);
        let carried_format = format(precision(&mut random), layout.clone());
        let mut graph = ComputeGraph::new();
        let carried = graph.host_input("state", [size, size]).unwrap();
        let weights = (0..count)
            .map(|index| graph.parameter(format!("weight.{index}"), [size, size]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let sequence = graph.value_sequence("weights", weights.clone()).unwrap();
        let output = graph
            .repeat(count, [carried], [], [sequence], |body, arguments| {
                Ok(vec![
                    body.gemm(arguments.carried[0], arguments.iterated[0])?,
                ])
            })
            .unwrap()[0];
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles).with_input(carried, carried_format.clone());
        for weight in weights {
            config
                .inputs
                .insert(weight, format(precision(&mut random), layout.clone()));
        }

        let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let repeat = lowered
            .operations
            .iter()
            .find_map(|operation| match &operation.kind {
                MidOperationKind::Repeat(repeat) => Some(repeat),
                _ => None,
            })
            .unwrap();
        assert_eq!(repeat.count, count, "random case {case}");
        assert_eq!(repeat.iterated_inputs.len(), 1, "random case {case}");
        assert_eq!(
            repeat.iterated_inputs[0].len(),
            count as usize,
            "random case {case}"
        );
        let sequence_format = &value(&lowered, repeat.iterated_inputs[0][0])
            .tensor_type
            .format;
        assert!(
            repeat.iterated_inputs[0].iter().all(|value_id| {
                &value(&lowered, *value_id).tensor_type.format == sequence_format
            })
        );
        assert_eq!(
            &value(&lowered, repeat.body.yields[0]).tensor_type.format,
            &carried_format,
            "random case {case}"
        );
        assert_eq!(
            &value(&lowered, lowered.outputs[0]).tensor_type.format,
            &carried_format,
            "random case {case}"
        );
        assert_conversions_are_explicit(&lowered, &lowered.operations);
        assert_conversions_are_explicit(&lowered, &repeat.body.operations);
    }
}

#[test]
fn randomized_single_use_views_are_claimed_by_slice_consumers() {
    let mut random = fastrand::Rng::with_seed(0x6465_6665_7272_6564);
    for case in 0..RANDOM_CASES / 32 {
        let heads = random.u32(2..=6);
        let head_width = random.u32(4..=40) * 2;
        let tokens = random.u32(1..=3) * AMP_INNER_BLOCK;
        let model_width = heads * head_width;
        let tiles = u16::try_from(heads * tokens.div_ceil(AMP_INNER_BLOCK)).unwrap();
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [1, tokens, model_width]).unwrap();
        let mut projected = Vec::new();
        let mut parameters = Vec::new();
        for index in 0..3 {
            let weights = graph
                .parameter(format!("projection.{index}"), [model_width, model_width])
                .unwrap();
            parameters.push(weights);
            projected.push(graph.gemm(input, weights).unwrap());
        }
        let split = projected
            .iter()
            .map(|&value| graph.split_heads(value, heads).unwrap())
            .collect::<Vec<_>>();
        let output = graph.flash_attention(split[0], split[1], split[2]).unwrap();
        graph.set_outputs([output]).unwrap();
        let mut config = PipelineConfig::new(tiles)
            .with_attention_products(AttentionProducts::SharedRows)
            .with_automatic_input(input, Precision::F16);
        for parameter in parameters {
            config = config.with_automatic_input(parameter, Precision::F16);
        }
        config.conversion_streaming = ConversionStreamingPolicy::Always;

        let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let producers = lowered
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::View(_),
                            ..
                        },
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(producers.len(), split.len(), "random case {case}");
        assert!(
            producers.iter().all(|operation| {
                operation.estimated_cycles == 0
                    && operation
                        .operator_plan()
                        .is_some_and(|plan| plan.deferred_output.is_some())
            }),
            "random case {case}"
        );
        let consumer = lowered
            .operations
            .iter()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::FlashAttention { .. },
                            ..
                        },
                        ..
                    }
                )
            })
            .unwrap();
        let claims = consumer.deferred_inputs();
        assert_eq!(claims.len(), split.len(), "random case {case}");
        assert!(claims.iter().all(Option::is_some), "random case {case}");
        let compact = implementation::resolve(lowered.clone()).unwrap();
        for claim in claims.iter().flatten() {
            assert!(
                compact
                    .operations
                    .iter()
                    .all(|op| !op.results.contains(&claim.producer)),
                "claimed view must not be separately materialized"
            );
        }

        assert!(
            compact
                .operations
                .iter()
                .all(|op| !matches!(op.kind, MidOperationKind::Operator { .. }))
        );
        assert!(compact.operations.iter().any(|op| matches!(
            op.kind,
            MidOperationKind::Primitive(Primitive::Copy {
                mapping: CoordinateMapping { view: Some(_), .. },
                ..
            })
        )));
        assert!(compact.operations.iter().any(|op| matches!(
            op.kind,
            MidOperationKind::Primitive(Primitive::Compute {
                kernel: TileKernelSpec::AttentionSoftmax { .. },
                ..
            })
        )));
        for op in &compact.operations {
            if matches!(
                op.kind,
                MidOperationKind::Primitive(Primitive::Compute {
                    kernel: TileKernelSpec::Gemm { .. } | TileKernelSpec::AttentionSoftmax { .. },
                    ..
                })
            ) {
                assert_eq!(
                    compact.values[op.results[0].index() as usize]
                        .tensor_type
                        .format
                        .precision,
                    Precision::F16
                );
            }
        }
        let program = expand_tiles(&lowered).unwrap();
        let cycles = crate::estimate::program_cycles(&program, None).unwrap();
        assert_eq!(program.estimated_cycles, cycles.total);
        assert_eq!(program.estimated_exchange_cycles, cycles.exchange);
        let tiled = crate::low::lower_to_tiles(&program, config.diagnostic_checkpoints);
        crate::KernelBuildPlan::from_program(&tiled)
            .unwrap_or_else(|error| panic!("random case {case}: {error}"));
        for run in &tiled.kernel_runs {
            assert_eq!(run.inputs.len(), run.requirements.inputs.len());
            for (operand, requirement) in run.inputs.iter().zip(&run.requirements.inputs) {
                assert_eq!(
                    requirement.format,
                    tiled.shards[operand.views[0].shard.index() as usize]
                        .tensor_type
                        .format
                );
            }
            assert_eq!(
                run.requirements.output.format,
                tiled.shards[run.output.shard.index() as usize]
                    .tensor_type
                    .format
            );
        }
        let attention_phases = tiled
            .exchange_phases
            .iter()
            .filter(|phase| phase.provenance.operation == consumer.source)
            .count();
        assert!(
            attention_phases > 0,
            "random case {case}: deferred movement must be priced"
        );
        assert!(
            attention_phases
                <= compact
                    .operations
                    .iter()
                    .filter(|op| op.source == consumer.source
                        && matches!(op.kind, MidOperationKind::Primitive(Primitive::Copy { .. })))
                    .count(),
            "random case {case}: {attention_phases} attention exchange phases"
        );
    }
}

#[test]
fn randomized_unclaimed_deferred_offers_restore_materialization_cost() {
    let mut random = fastrand::Rng::with_seed(0x756e_636c_6169_6d65);
    for case in 0..RANDOM_CASES / 8 {
        let batch = random.u32(1..=4);
        let heads = random.u32(1..=8);
        let rows = random.u32(1..=4) * AMP_INNER_BLOCK;
        let head_width = random.u32(1..=4) * AMP_COLUMN_MICRO;
        let mut graph = ComputeGraph::new();
        let input = graph
            .host_input("input", [batch, rows, heads * head_width])
            .unwrap();
        let output = graph.split_heads(input, heads).unwrap();
        graph.set_outputs([output]).unwrap();
        let tiles = u16::try_from(batch * heads).unwrap();
        let config = PipelineConfig::new(tiles).with_automatic_input(input, Precision::F16);

        let lowered = lower(&graph, &config, &Ipu21CostModel).unwrap();
        let operation = lowered
            .operations
            .iter()
            .find(|operation| {
                matches!(
                    operation.kind,
                    MidOperationKind::Operator {
                        plan: OperatorPlan {
                            operator: MidOperator::View(_),
                            ..
                        },
                        ..
                    }
                )
            })
            .unwrap();
        assert!(operation.operator_plan().unwrap().deferred_output.is_none());
        assert!(operation.estimated_cycles != 0, "random case {case}");
    }
}

#[test]
fn operator_shortlists_stay_bounded_when_format_diversity_exceeds_width() {
    let inputs = [
        TensorType::new([128, 64], Precision::F16, Layout::row_sharded(64)),
        TensorType::new([64, 64], Precision::F16, Layout::row_sharded(64)),
    ];
    let candidates = default_operator_candidates(64)
        .into_iter()
        .filter_map(|candidate| match candidate {
            OperatorCandidate::Concrete(concrete) => Some(concrete.plan),
            _ => None,
        })
        .filter(|plan| matches!(plan.operator, MidOperator::Gemm { .. }))
        .collect::<Vec<_>>();
    assert!(
        candidates
            .iter()
            .map(operator_candidate_compatibility)
            .collect::<BTreeSet<_>>()
            .len()
            > 2
    );
    let selected = retain_operator_candidates(
        candidates,
        &inputs,
        &TensorShape::new([128, 64]),
        &Ipu21CostModel,
        2,
    );
    assert!((2..=4).contains(&selected.len()));
}

#[test]
fn selected_mid_size_is_independent_of_tile_count() {
    let mut sizes = Vec::new();
    for tiles in [1, 4, 64, 1024] {
        let mut graph = ComputeGraph::new();
        let input = graph.host_input("input", [1024, 64]).unwrap();
        let output = graph.gelu(input).unwrap();
        graph.set_outputs([output]).unwrap();
        let config = PipelineConfig::new(tiles)
            .with_active_tile_counts([tiles])
            .with_input(input, format(Precision::F16, Layout::row_sharded(tiles)));
        let recipe = lower(&graph, &config, &Ipu21CostModel).unwrap();
        assert!(
            crate::low::expand::expand_tiles(&recipe, true).is_err(),
            "low must reject unresolved operator recipes"
        );
        let selected = implementation::resolve(recipe.clone()).unwrap();
        assert!(selected.operations.iter().all(|op| matches!(
            op.kind,
            MidOperationKind::Primitive(_) | MidOperationKind::Convert(_)
        )));
        let mut fresh = recipe.clone();
        for op in &mut fresh.operations {
            if let MidOperationKind::Operator { implementation, .. } = &mut op.kind {
                *implementation = None;
            }
        }
        assert_eq!(selected, implementation::resolve(fresh.clone()).unwrap());
        assert!(crate::estimate::analyze_mid(&selected, &BTreeMap::new()).is_some());
        sizes.push((selected.values.len(), selected.operations.len()));
    }
    assert!(sizes.windows(2).all(|pair| pair[0] == pair[1]), "{sizes:?}");
}

#[test]
fn uneven_mlp_products_preserve_global_coordinates() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [1, 729, 1152]).unwrap();
    let up = graph.parameter("up", [1, 1152, 4304]).unwrap();
    let down = graph.parameter("down", [1, 4304, 1152]).unwrap();
    let hidden = graph.gemm(input, up).unwrap();
    let hidden = graph.gelu(hidden).unwrap();
    let output = graph.gemm(hidden, down).unwrap();
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(1472)
        .with_automatic_input(input, Precision::F16)
        .with_automatic_input(up, Precision::F16)
        .with_automatic_input(down, Precision::F16);
    let mid = super::lower_finalists(&graph, &config, &Ipu21CostModel, 1)
        .unwrap()
        .remove(0);
    let tiles = crate::low::expand::expand_tiles(&mid, true).unwrap();
    let useful: u64 = tiles
        .kernel_runs
        .iter()
        .filter_map(|run| run.product_flops)
        .map(|f| f[0])
        .sum();
    assert_eq!(useful, 4 * 729 * 1152 * 4304);
    for run in &tiles.kernel_runs {
        if !matches!(run.kernel, TileKernelSpec::Gemm { .. }) {
            continue;
        }
        let output = &run.output.extents;
        let left = &run.inputs[0].views[0].extents;
        let right = &run.inputs[1].views[0].extents;
        let bounds = |e: &ShardExtent| (e.start, e.logical_end, e.physical_end);
        assert_eq!(
            bounds(&left[left.len() - 2]),
            bounds(&output[output.len() - 2]),
            "left/output rows {run:?}"
        );
        assert_eq!(
            bounds(&left[left.len() - 1]),
            bounds(&right[right.len() - 2]),
            "inner {run:?}"
        );
        assert_eq!(
            bounds(&right[right.len() - 1]),
            bounds(&output[output.len() - 1]),
            "right/output columns {run:?}"
        );
    }
}

#[test]
fn materialized_attention_packs_values_for_the_full_product() {
    let mut graph = ComputeGraph::new();
    let query = graph.host_input("query", [4, 17, 72]).unwrap();
    let key = graph.host_input("key", [4, 73, 72]).unwrap();
    let value = graph.host_input("value", [4, 73, 72]).unwrap();
    let output = graph.flash_attention(query, key, value).unwrap();
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(64)
        .with_attention_strategy(AttentionStrategy::Materialized)
        .with_attention_products(AttentionProducts::SharedRows)
        .with_automatic_input(query, Precision::F16)
        .with_automatic_input(key, Precision::F16)
        .with_automatic_input(value, Precision::F16);
    let mid = super::lower_finalists(&graph, &config, &Ipu21CostModel, 1)
        .unwrap()
        .remove(0);
    let product = mid
        .operations
        .iter()
        .find(|op| {
            matches!(
                op.kind,
                MidOperationKind::Primitive(Primitive::Compute {
                    kernel: TileKernelSpec::Gemm {
                        inner_block: 128,
                        ..
                    },
                    product: Some(ProductAxes {
                        right_inner: TensorAxis::FromEnd(2),
                        ..
                    }),
                    ..
                })
            )
        })
        .unwrap();
    assert_eq!(
        mid.values[product.inputs[1].index() as usize]
            .tensor_type
            .format
            .layout
            .order,
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: 128,
            column_block: 16
        })
    );
    assert!(mid.values.iter().any(|value| {
        value.tensor_type.format.layout.order == ElementOrder::Amp(AmpOrder::TransposedRight)
            && value
                .tensor_type
                .format
                .layout
                .tiling
                .axes
                .iter()
                .any(|axis| {
                    axis.axis.resolve(3) == Ok(1)
                        && axis.partitions == 2
                        && axis.block_size == AMP_INNER_BLOCK
                })
    }));
}

#[test]
fn shortlist_prices_execution_instead_of_boundary_storage() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [1, 729, 1152]).unwrap();
    let weight = graph.parameter("weight", [1, 1152, 4304]).unwrap();
    let output = graph.gemm(input, weight).unwrap();
    graph.set_outputs([output]).unwrap();
    let mut plans = Vec::new();
    for (r, c, k) in [(4, 92, 4), (3, 27, 18)] {
        let config = PipelineConfig::new(1472)
            .with_automatic_input(input, Precision::F16)
            .with_automatic_input(weight, Precision::F16)
            .with_gemm_plan_constraint(GemmPlanConstraint {
                source_operation: 0,
                orientation: GemmOrientation::Normal,
                row_partitions: r,
                column_partitions: c,
                inner_partitions: k,
                result_row_partitions: k,
                result_column_partitions: 1,
                output_column_block: if k == 4 { 48 } else { 160 },
                weight_memory_class: MemoryClass::Ipu21Interleaved,
                reduction_staging: ReductionStaging::Complete,
                local_weight_staging: LocalOperandStaging::Direct,
            });
        let mid = lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        plans.push(
            mid.operations
                .iter()
                .find_map(|op| op.operator_plan())
                .unwrap()
                .clone(),
        );
    }
    let inputs = [
        TensorType::new([1, 729, 1152], Precision::F16, Layout::row_sharded(1472)),
        TensorType::new([1, 1152, 4304], Precision::F16, Layout::row_sharded(1472)),
    ];
    let output = TensorShape::new([1, 729, 4304]);
    let boundary_bytes = |plan: &OperatorPlan| {
        let (inputs, output) = plan.tensor_types(&inputs, &output);
        inputs
            .iter()
            .chain(std::iter::once(&output))
            .map(|t| crate::estimate::tensor_memory(t).total())
            .sum::<u64>()
    };
    assert!(boundary_bytes(&plans[0]) > boundary_bytes(&plans[1]));
    let expected = plans[0].clone();
    let selected = retain_operator_candidates(plans, &inputs, &output, &Ipu21CostModel, 1);
    assert_eq!(selected, vec![expected]);
}

#[test]
fn unconstrained_mlp_shortlists_preserve_historical_memory_alternatives() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("input", [1, 729, 1152]).unwrap();
    let up = graph.parameter("up", [1, 1152, 4304]).unwrap();
    let down = graph.parameter("down", [1, 4304, 1152]).unwrap();
    let hidden = graph.gemm(input, up).unwrap();
    let hidden = graph.gelu(hidden).unwrap();
    let output = graph.gemm(hidden, down).unwrap();
    graph.set_outputs([output]).unwrap();
    let mut config = PipelineConfig::new(1472);
    for tiles in shape_aware_active_tile_counts(1472, graph.value_shapes().values()) {
        for candidate in operator_candidates_for_tile_count(tiles) {
            if !config.operator_candidates.contains(&candidate) {
                config.operator_candidates.push(candidate);
            }
        }
    }
    let costs = MemoizedCostModel::new(&Ipu21CostModel, 1472);
    for (operation, inner, columns, grid) in
        [(0, 1152, 4304, (4, 92, 4)), (2, 4304, 1152, (4, 24, 15))]
    {
        let inputs = [
            TensorType::new([1, 729, inner], Precision::F16, Layout::row_sharded(1472)),
            TensorType::new(
                [1, inner, columns],
                Precision::F16,
                Layout::row_sharded(1472),
            ),
        ];
        let shape = TensorShape::new([1, 729, columns]);
        let generated = plans(
            &graph.operations()[operation],
            &inputs,
            &[false, true],
            &shape,
            &config,
            &costs,
            true,
            None,
            &[],
            &[],
        );
        let retained = retain_operator_candidates(
            generated,
            &inputs,
            &shape,
            &costs,
            config.planning_beam_width,
        );
        for memory in [MemoryClass::Ipu21Standard, MemoryClass::Ipu21Interleaved] {
            let expected = GemmPlanConstraint {
                source_operation: operation as u32,
                orientation: GemmOrientation::Normal,
                row_partitions: grid.0,
                column_partitions: grid.1,
                inner_partitions: grid.2,
                result_row_partitions: grid.2,
                result_column_partitions: 1,
                output_column_block: 48,
                weight_memory_class: memory,
                reduction_staging: ReductionStaging::Complete,
                local_weight_staging: LocalOperandStaging::Direct,
            };
            assert!(
                retained.iter().any(|plan| gemm_plan_matches(
                    &expected,
                    &plan.dispatch,
                    &plan.requirements.inputs
                )),
                "operation {operation}: lost {memory:?} historical geometry"
            );
        }
        assert!(retained.len() <= 2 * config.planning_beam_width);
        assert!(retained.iter().all(|plan| {
            let (inputs, output) = plan.tensor_types(&inputs, &shape);
            costs
                .implementation(plan, &inputs, &output)
                .unwrap()
                .peak_memory
                .standard_contiguous_overflow()
                == 0
        }));
    }
    // Do not pin complete-beam membership to the historical kernel timings:
    // changes to compute throughput can legitimately favor different grids.
}

#[test]
fn parallel_gemm_family_does_not_depend_on_concrete_templates() {
    let tiles = 16;
    let mut graph = ComputeGraph::new();
    let left = graph.host_input("left", [16, 128]).unwrap();
    let right = graph.parameter("right", [128, 128]).unwrap();
    let output = graph.gemm(left, right).unwrap();
    graph.set_outputs([output]).unwrap();
    let inputs = [
        TensorType::new([16, 128], Precision::F16, Layout::row_sharded(tiles)),
        TensorType::new([128, 128], Precision::F16, Layout::row_sharded(tiles)),
    ];
    let mut config = PipelineConfig::new(tiles);
    let generate = |config: &PipelineConfig| {
        plans(
            &graph.operations()[0],
            &inputs,
            &[false, true],
            &TensorShape::new([16, 128]),
            config,
            &Ipu21CostModel,
            true,
            None,
            &[],
            &[],
        )
    };
    config.operator_candidates = vec![OperatorCandidate::parallel_gemm(tiles)];
    let parallel = generate(&config);
    assert!(!parallel.is_empty());
    assert!(parallel.iter().all(|plan| matches!(
        plan.dispatch,
        OperatorDispatch::BlockedGemm {
            distribution: GemmDistribution::ParallelReduction { .. },
            ..
        }
    )));
    for orientation in [GemmOrientation::Normal, GemmOrientation::Swapped] {
        let plan = parallel.iter().find(|plan| matches!(plan.dispatch,
            OperatorDispatch::BlockedGemm { orientation: actual, .. } if actual == orientation
        )).expect("both GEMM orientations remain available");
        let (left, right) = orientation.operand_indices();
        assert_eq!(plan.requirements.inputs[left].access_tail_bytes, 16);
        assert_eq!(plan.requirements.inputs[right].access_tail_bytes, 0);
        assert_eq!(
            plan.requirements.distinct_elements,
            vec![vec![
                MemoryOperand::Output,
                MemoryOperand::Input(left as u16)
            ]]
        );
    }
    // The former C64 trigger now offers only the explicitly configured plan.
    config.operator_candidates = vec![amp_gemm_operator_candidate(
        Precision::F16,
        64,
        16,
        64,
        tiles,
    )]
    .into_iter()
    .map(OperatorCandidate::Concrete)
    .collect();
    let concrete = generate(&config);
    assert!(!concrete.is_empty());
    assert!(concrete.iter().all(|plan| matches!(
        plan.dispatch,
        OperatorDispatch::BlockedGemm {
            distribution: GemmDistribution::OutputStationary,
            ..
        }
    )));
}

#[test]
fn value_projection_retains_head_grouped_swapped_output_from_packed_activations() {
    let mut graph = ComputeGraph::new();
    let x = graph.host_input("x", [1, 729, 1152]).unwrap();
    let w = graph.parameter("w", [1, 1152, 1152]).unwrap();
    let projection = graph.gemm(x, w).unwrap();
    let heads = graph.split_heads(projection, 16).unwrap();
    let result = graph.flash_attention(heads, heads, heads).unwrap();
    graph.set_outputs([result]).unwrap();
    let config = PipelineConfig::new(1472).with_planning_beam_width(4);
    let demands = OutputDemands::new(graph.operations(), graph.value_shapes(), &config);
    let uses = BTreeMap::from([(projection, 1)]);
    let mut search = CandidateSearch::new(
        &graph.operations()[0],
        &graph.operations()[1..],
        false,
        &uses,
        &config,
        &demands,
    );
    let inputs = [
        TensorType::new([1, 729, 1152], Precision::F16, Layout::amp_left(64, 1472)),
        TensorType::new([1, 1152, 1152], Precision::F16, Layout::row_sharded(1472)),
    ];
    let candidates = search
        .generate(
            &inputs,
            &[false, true],
            &[false, true],
            graph.value_shape(projection).unwrap(),
            &Ipu21CostModel,
        )
        .unwrap();
    let requested = OutputDemand {
        order: ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: 64,
            column_block: 16,
        }),
        column_groups: 16,
        inner_grain: 1,
    };
    assert!(
        candidates.iter().any(|plan| requested.matches(
            &plan.requirements.output.format.layout,
            graph.value_shape(projection).unwrap()
        )),
        "missing V-compatible output among {} candidates",
        candidates.len()
    );
    assert!(candidates.iter().any(|plan| matches!(
        plan.dispatch,
        OperatorDispatch::BlockedGemm {
            orientation: GemmOrientation::Normal,
            ..
        }
    )));
}

#[test]
fn beam_reserves_requested_formats_before_incidental_layout_diversity() {
    let mut graph = ComputeGraph::new();
    let projection = graph.host_input("projection", [1, 17, 288]).unwrap();
    let heads = graph.split_heads(projection, 4).unwrap();
    graph.flash_attention(heads, heads, heads).unwrap();
    let config = PipelineConfig::new(4);
    let demands = OutputDemands::new(graph.operations(), graph.value_shapes(), &config);
    let mut compatible = Layout::row_major(TensorTiling::sharded(TensorAxis::FromEnd(1), 4));
    compatible.order = ElementOrder::Amp(AmpOrder::TransposedLeft);
    let requested = OutputDemand {
        order: compatible.order,
        column_groups: 4,
        inner_grain: 1,
    };
    let shape = graph.value_shape(projection).unwrap();
    assert!(requested.matches(&compatible, shape));
    let branches = [Layout::row_sharded(1), Layout::amp_left(64, 1), compatible]
        .into_iter()
        .zip([1, 2, 10])
        .map(|(layout, total)| {
            let mut state = LoweringState::default();
            let id = state.value(
                projection,
                TensorType::new(shape.0.clone(), Precision::F16, layout),
            );
            BeamBranch {
                values: BTreeMap::from([(projection, id)]),
                state,
                operations: Vec::new(),
                peak_memory: MemoryPeaks::default(),
                analysis: std::sync::OnceLock::from(Some((
                    crate::estimate::ProgramCycles { total, exchange: 0 },
                    MemoryPeaks::default(),
                    true,
                ))),
            }
        })
        .collect();
    let (retained, _, _, _) = retain_pareto_beam(
        branches,
        &BTreeSet::from([projection]),
        &RegionPlanningConstraints::default(),
        &Ipu21CostModel,
        2,
        &demands,
    );
    assert_eq!(retained.len(), 2);
    assert!(retained.iter().any(|branch| {
        requested.matches(
            &branch
                .state
                .get(branch.values[&projection])
                .tensor_type
                .format
                .layout,
            shape,
        )
    }));
    assert_eq!(
        deferred_aware_branch_score(&retained[0], &BTreeSet::new()),
        1
    );
}

#[test]
fn generic_view_distributes_matrix_rows_instead_of_whole_batches() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("heads", [16, 729, 72]).unwrap();
    let output = graph.view(input, AxisFactorView::new(0, 2, 16)).unwrap();
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(1472).with_automatic_input(input, Precision::F16);
    let program = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let output = &program.values[program.outputs[0].index() as usize];
    assert!(output.tensor_type.format.layout.tiling.tile_count > 1);
}

#[test]
fn automatic_repeat_state_keeps_an_unreplicated_boundary() {
    let mut graph = ComputeGraph::new();
    let input = graph.host_input("state", [1, 17, 64]).unwrap();
    let weight = graph.parameter("weight", [1, 64, 64]).unwrap();
    let output = graph
        .repeat(3, [input], [weight], [], |body, args| {
            Ok(vec![body.gemm(args.carried[0], args.invariants[0])?])
        })
        .unwrap()[0];
    graph.set_outputs([output]).unwrap();
    let config = PipelineConfig::new(64)
        .with_automatic_input(input, Precision::F16)
        .with_automatic_input(weight, Precision::F16);
    let mid = lower(&graph, &config, &Ipu21CostModel).unwrap();
    let repeat = mid
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            MidOperationKind::Repeat(repeat) => Some((op, repeat)),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        mid.values[repeat.1.body.arguments[0].index() as usize]
            .tensor_type
            .format
            .layout
            .tiling
            .replicas,
        1
    );
    for (input, argument) in repeat.0.inputs.iter().zip(&repeat.1.body.arguments) {
        assert_eq!(
            mid.values[input.index() as usize].tensor_type,
            mid.values[argument.index() as usize].tensor_type
        );
    }
    crate::expand_tiles(&mid).unwrap();
}

#[test]
fn attention_profile_flops_exclude_scratch_padding_and_key_tails() {
    for (strategy, products) in [
        (AttentionStrategy::Flash, AttentionProducts::SharedRows),
        (
            AttentionStrategy::Materialized,
            AttentionProducts::SharedRows,
        ),
        (AttentionStrategy::Materialized, AttentionProducts::QkOnly),
        (AttentionStrategy::Materialized, AttentionProducts::PvOnly),
        (
            AttentionStrategy::Materialized,
            AttentionProducts::Independent,
        ),
    ] {
        let mut graph = ComputeGraph::new();
        let q = graph.host_input("q", [4, 17, 72]).unwrap();
        let k = graph.host_input("k", [4, 73, 72]).unwrap();
        let v = graph.host_input("v", [4, 73, 72]).unwrap();
        let result = graph.flash_attention(q, k, v).unwrap();
        graph.set_outputs([result]).unwrap();
        let config = PipelineConfig::new(64)
            .with_attention_strategy(strategy)
            .with_attention_products(products)
            .with_automatic_input(q, Precision::F16)
            .with_automatic_input(k, Precision::F16)
            .with_automatic_input(v, Precision::F16);
        let mid = super::lower_finalists(&graph, &config, &Ipu21CostModel, 1)
            .unwrap()
            .remove(0);
        assert_eq!(
            mid.values[mid.outputs[0].index() as usize]
                .tensor_type
                .format
                .precision,
            Precision::F16
        );
        let tiles = crate::low::expand::expand_tiles(&mid, true).unwrap();
        let tiled = crate::low::lower_to_tiles(&tiles, false);
        crate::KernelBuildPlan::from_program(&tiled).unwrap();
        for phase in &tiles.exchange_phases {
            for transfer in &phase.transfers {
                let source = &tiles.shards[transfer.source.shard.index() as usize];
                let spans = match transfer.span_order(&tiles.shards) {
                    crate::CopyOrder::Physical => crate::view_byte_spans(source, &transfer.source),
                    crate::CopyOrder::Semantic => {
                        crate::logical_view_byte_spans(source, &transfer.source)
                    }
                }
                .unwrap();
                assert!(
                    spans.iter().all(|span| span.bytes.is_multiple_of(4)),
                    "half-word transfer: {products:?} {transfer:?}"
                );
            }
        }
        let mut total = [0u64; 2];
        for flops in tiles.kernel_runs.iter().filter_map(|run| run.product_flops) {
            assert!(flops[0] <= flops[1]);
            total[0] += flops[0];
            total[1] += flops[1];
        }
        assert_eq!(total[0], 4 * 4 * 17 * 73 * 72, "{strategy:?}/{products:?}");
        assert!(total[0] < total[1]);
    }
}
