use super::*;
use crate::mid::Compute;
use crate::*;

fn iterated_sum(partials: u16) -> (LowProgram, Placement) {
    let id = MidValueId::from_index;
    let values = (0..7)
        .map(|i| {
            let parameter = [1, 2, 4].contains(&i);
            MidValue {
                id: id(i),
                origin: ValueId::from_index(i),
                storage_group: id(i),
                tile_offset: 0,
                tensor_type: TensorType::new(
                    if parameter {
                        vec![u32::from(partials), 8, 16]
                    } else {
                        vec![8, 16]
                    },
                    Precision::F16,
                    if parameter {
                        Layout::row_major(TensorTiling::sharded(TensorAxis::FromStart(0), partials))
                    } else {
                        Layout::row_sharded(1)
                    },
                ),
            }
        })
        .collect();
    let mid = MidProgram {
        tile_count: partials,
        values,
        outputs: vec![id(6)],
        inputs: [0, 1, 2]
            .map(|i| MidInput {
                value: id(i),
                name: format!("input.{i}"),
                kind: if i == 0 {
                    GraphInputKind::Host
                } else {
                    GraphInputKind::Parameter
                },
            })
            .to_vec(),
        operations: vec![MidOperation {
            source: None,
            inputs: vec![id(0)],
            results: vec![id(6)],
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
            kind: MidOperationKind::Repeat(MidRepeat {
                count: 2,
                carried_inputs: 1,
                invariant_inputs: 0,
                iterated_inputs: vec![vec![id(1), id(2)]],
                body: MidRegion {
                    arguments: vec![id(3), id(4)],
                    yields: vec![id(5)],
                    estimated_cycles: 0,
                    peak_memory: Default::default(),
                    operations: vec![MidOperation {
                        source: None,
                        inputs: vec![id(4)],
                        results: vec![id(5)],
                        estimated_cycles: 0,
                        estimated_exchange_cycles: 0,
                        kind: MidOperationKind::Compute(Compute::Sum {
                            axis: 0,
                            staging: ReductionStaging::Complete,
                        }),
                    }],
                },
            }),
        }],
        ..MidProgram::default()
    };
    let graph = crate::low::expand::expand_tiles(&mid, false).unwrap();
    let low = crate::low::lower_to_tiles(&graph, false);
    let placement = place(&low).unwrap();
    (low, placement)
}

#[test]
fn sum_aliases_follow_iterated_parameters_in_local_copies_and_exchanges() {
    for partials in [1, 2] {
        let (low, placement) = iterated_sum(partials);
        let kernels = KernelBuildPlan::from_program(&low).unwrap();
        let phases = crate::exchange::lower_exchanges(
            &low,
            &placement,
            &ipu_exchange::Topology::c600(),
            false,
        )
        .unwrap()
        .phases;
        let lowering =
            TileProgramLowering::new(&low, &placement, &phases, &kernels, 0x100, partials, false)
                .unwrap();
        let mut moving_reads = 0;
        let mut moving_sends = 0;
        for tile in 0..partials {
            let program = lowering.lower_tile(tile).unwrap();
            let TileStep::Repeat(repeat) = &program.steps[0] else {
                panic!("repeat")
            };
            assert_eq!(repeat.iterated_pointers.len(), 1);
            assert_eq!(repeat.iterated_pointers[0].stride_bytes, 256);
            for step in &repeat.body {
                match step {
                    TileStep::Compute(compute)
                        if compute.symbol
                            == if partials == 1 {
                                COPY_U64_SYMBOL
                            } else {
                                "reduce_sum_f16"
                            }
                            && matches!(
                                compute.input_addresses[0],
                                TileAddress::RepeatPointer {
                                    index: 0,
                                    offset: 0
                                }
                            ) =>
                    {
                        moving_reads += 1
                    }
                    TileStep::Exchange(exchange)
                        if matches!(
                            exchange.outgoing_base,
                            Some(TileAddress::RepeatPointer {
                                index: 0,
                                offset: 0
                            })
                        ) =>
                    {
                        moving_sends += 1
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(
            moving_reads, 1,
            "the locally owned partial must change each iteration"
        );
        assert_eq!(
            moving_sends,
            partials - 1,
            "remote partials must also change each iteration"
        );
    }
}

#[test]
fn repeat_pointers_use_complete_placed_access_requirements() {
    let (low, _) = iterated_sum(2);
    for count in [1, 2] {
        let mut graph = (*low.program).clone();
        for run in &mut graph.kernel_runs {
            for input in &mut std::sync::Arc::make_mut(&mut run.metadata)
                .requirements
                .inputs
            {
                input.alignment = 64;
                input.access_tail_bytes = 96;
            }
        }
        let BlockOperation::Repeat(repeat) = &mut graph.body.operations[0] else {
            panic!("repeat")
        };
        repeat.count = count;
        for binding in &mut repeat.bindings {
            for sequence in &mut binding.iterated {
                sequence.inputs.truncate(count as usize);
            }
        }
        let low = lower_to_tiles(&std::sync::Arc::new(graph), false);
        // The same sequence contract must work in either physical region.
        for base in [
            crate::memory::IPU21_DATA_BASE,
            ipu_package::IPU21_INTERLEAVED_MEMORY_BASE,
        ] {
            let placement = crate::place::place_with_ranges(
                &low,
                &[(base, ipu_package::IPU21_APPLICATION_MEMORY_LIMIT)],
            )
            .unwrap();
            let sequence = &low.repeat_runs[0].binding.iterated[0];
            let stride = placement.sequence_strides[&sequence.argument];
            assert!(stride >= 256 + 96 && stride.is_multiple_of(64));
            for (index, shard) in sequence.inputs.iter().enumerate() {
                assert_eq!(
                    placement.shard_addresses[shard],
                    placement.shard_addresses[&sequence.argument] + index as u32 * stride
                );
            }
            let kernels = KernelBuildPlan::from_program(&low).unwrap();
            let phases = crate::exchange::lower_exchanges(
                &low,
                &placement,
                &ipu_exchange::Topology::c600(),
                false,
            )
            .unwrap()
            .phases;
            let lowering =
                TileProgramLowering::new(&low, &placement, &phases, &kernels, 0x100, 2, false)
                    .unwrap();
            let program = lowering.lower_tile(0).unwrap();
            let TileStep::Repeat(repeat) = &program.steps[0] else {
                panic!("repeat")
            };
            assert_eq!(repeat.iterated_pointers[0].stride_bytes, stride);
        }
    }
}

#[test]
fn repeat_sequences_preserve_regular_offsets_between_shifted_aliases() {
    let (low, _) = iterated_sum(1);
    let sequence = &low.repeat_runs[0].binding.iterated[0];
    let mut graph = (*low.program).clone();
    for (index, shard) in sequence.inputs.iter().enumerate() {
        let mut root = graph.shards[shard.index() as usize].clone();
        root.id = BlockValueId::from_index(graph.shards.len() as u32);
        root.definition = ShardDefinition::Staging;
        graph.shards[shard.index() as usize].definition = ShardDefinition::ShiftedAlias {
            source: root.id,
            offset: index as i32 * 64,
        };
        graph.shards.push(root);
    }
    let projected = lower_to_tiles(&std::sync::Arc::new(graph), false);
    let placement = place(&projected).unwrap();
    let stride = placement.sequence_strides[&sequence.argument];
    let first = placement.shard_addresses[&sequence.argument];
    for (index, shard) in sequence.inputs.iter().enumerate() {
        assert_eq!(
            placement.shard_addresses[shard],
            first + index as u32 * stride
        );
    }
}

#[test]
fn pointer_resolution_preserves_signed_offsets_through_alias_chains() {
    let (low, mut placement) = iterated_sum(1);
    let argument = low.repeat_runs[0].binding.iterated[0].argument;
    let mut shards = low.shards.clone();
    let overrides = BTreeMap::from([(
        argument,
        TileAddress::RepeatPointer {
            index: 0,
            offset: 32,
        },
    )]);
    let mut source = argument;
    let mut displacement = 0;
    for delta in [None, None, Some(-32768), Some(65536)] {
        let mut alias = shards[source.index() as usize].clone();
        alias.id = BlockValueId::from_index(shards.len() as u32);
        alias.definition = match delta {
            Some(offset) => ShardDefinition::ShiftedAlias { source, offset },
            None if source == argument => ShardDefinition::Alias(source),
            None => ShardDefinition::WritableAlias(source),
        };
        displacement += delta.unwrap_or(0);
        placement.shard_addresses.insert(
            alias.id,
            placement.shard_addresses[&argument]
                .checked_add_signed(displacement)
                .unwrap(),
        );
        source = alias.id;
        shards.push(alias);
        assert_eq!(
            crate::kernel::resolve_shard_address(
                &shards,
                &placement.shard_addresses,
                &overrides,
                source
            )
            .unwrap(),
            TileAddress::RepeatPointer {
                index: 0,
                offset: displacement + 32
            }
        );
        assert_eq!(
            crate::kernel::resolve_shard_address(
                &shards,
                &placement.shard_addresses,
                &BTreeMap::new(),
                source
            )
            .unwrap(),
            TileAddress::Absolute(placement.shard_addresses[&source])
        );
    }
    let output = BlockValueId::from_index(shards.len() as u32);
    let mut block = shards[source.index() as usize].clone();
    block.id = output;
    block.definition = ShardDefinition::Staging;
    shards.push(block);
    placement.shard_addresses.insert(output, 0x80000);
    let view = |shard: BlockValueId| ShardView {
        shard,
        extents: shards[shard.index() as usize].extents.clone(),
    };
    let kernel = TileKernelSpec::Gelu;
    let run = KernelRun::new(
        low.repeat_runs[0].provenance,
        kernel.clone(),
        vec![KernelOperand {
            views: vec![view(source)],
        }],
        vec![view(output)],
        KernelRequirements::new(
            &kernel,
            [shards[source.index() as usize].tensor_type.format.clone()],
            vec![shards[output.index() as usize].tensor_type.format.clone()],
        ),
    );
    let call = materialize_kernel_run(
        &run,
        &shards,
        &placement.shard_addresses,
        &KernelBuildPlan::default(),
        &overrides,
    )
    .unwrap();
    assert_eq!(
        call.input_addresses,
        [TileAddress::RepeatPointer {
            index: 0,
            offset: displacement + 32
        }]
    );
    let mut code = crate::TileCode::default();
    crate::emit_address(
        &mut code,
        3,
        TileAddress::RepeatPointer {
            index: 0,
            offset: -32768,
        },
        Some(1),
    )
    .unwrap();
    assert_eq!(code.words.len(), 2);
    assert_eq!(
        code.words[1],
        ipu_exchange::encode_add_m_immediate(3, 3, -32768).unwrap()
    );
}
