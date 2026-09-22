//! Remove padding clears using resolved kernel access contracts and graph provenance.
//! Finite scratch is valid only in an all-F16 arena initialized once at load time.

use crate::CopyOrder;
use crate::kernel::PaddingRequirement;
use crate::low::storage::storage_location;
use crate::low::*;
use crate::mid::MidOperationKind;
use crate::tensor::Precision;
use ipu_target::Target;
use std::collections::BTreeSet;

#[tracing::instrument(skip_all)]
pub(super) fn eliminate(target: Target, program: &mut TileGraph) -> ExpansionResult<()> {
    let uses = crate::low::uses::StorageUses::analyze(program);
    let root = |id: BlockValueId| uses.roots[id.index() as usize];
    let mut clears = vec![Vec::new(); uses.allocations.len()];
    let mut removable = vec![false; program.kernel_runs.len()];
    let all_f16 = program
        .shards
        .iter()
        .all(|shard| shard.tensor_type.format.precision == Precision::F16);
    let mut parameter_storage = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        .flat_map(|input| {
            program
                .value_views(input.value)
                .iter()
                .map(|view| view.shard)
        })
        .filter(|&id| id.index() as usize == root(id) && uses.allocations[root(id)].read_only)
        .map(root)
        .collect::<BTreeSet<_>>();
    // Parameter origin alone says nothing about the destination's padding.
    // Propagate the proof only through complete, identical physical layouts.
    // Partial/shifted/rearranged writes require a finer proof and remain unknown.
    let preserves_padding = |source: BlockValueId, destination: BlockValueId| {
        let a = &program.shards[source.index() as usize];
        let b = &program.shards[destination.index() as usize];
        source.index() as usize == root(source)
            && destination.index() as usize == root(destination)
            && !uses.allocations[root(destination)].boundary
            && a.extents == b.extents
            && a.tensor_type.format.precision == b.tensor_type.format.precision
            && a.tensor_type.format.layout.order == b.tensor_type.format.layout.order
    };
    let mut incoming = std::collections::BTreeMap::<usize, BTreeSet<usize>>::new();
    for operation in program.body.walk() {
        match operation {
            BlockOperation::Copy { copy, .. } => {
                let copy = program.local_copies[copy.0 as usize].movement();
                incoming.entry(root(copy.destination)).or_default().insert(
                    if preserves_padding(copy.source, copy.destination)
                        && copy.source_offset == 0
                        && copy.destination_offset == 0
                        && copy.pattern == CopyPattern::Contiguous
                        && copy.bytes
                            == crate::low::storage::shard_storage_bytes(
                                &program.shards[copy.destination.index() as usize],
                            )?
                    {
                        root(copy.source)
                    } else {
                        root(copy.destination)
                    },
                );
            }
            BlockOperation::Exchange(id) => {
                for transfer in &program.exchange_phases[id.index() as usize].transfers {
                    for destination in &transfer.destinations {
                        incoming.entry(root(destination.shard)).or_default().insert(
                            if preserves_padding(transfer.source.shard, destination.shard)
                                && transfer.source.extents
                                    == program.shards[transfer.source.shard.index() as usize]
                                        .extents
                                && destination.extents
                                    == program.shards[destination.shard.index() as usize].extents
                            {
                                root(transfer.source.shard)
                            } else {
                                root(destination.shard)
                            },
                        );
                    }
                }
            }
            BlockOperation::Compute { run: id, .. } => {
                let run = &program.kernel_runs[id.0 as usize];
                if matches!(
                    run.kernel,
                    MidOperationKind::FillZero {
                        padding_only: true,
                        ..
                    }
                ) {
                    let backing = root(run.outputs[0].shard);
                    if !uses.allocations[backing].non_kernel_read {
                        clears[backing].push(*id);
                        removable[id.0 as usize] = true;
                    }
                }
                if !matches!(run.kernel, MidOperationKind::FillZero { .. }) {
                    // Arithmetic results are not known-zero-padded parameters.
                    for output in run.outputs.iter() {
                        incoming
                            .entry(root(output.shard))
                            .or_default()
                            .insert(root(output.shard));
                    }
                }
            }
            _ => {}
        }
    }
    if !removable.iter().any(|&remove| remove) {
        return Ok(());
    }
    for (&root, sources) in &incoming {
        if sources.contains(&root) {
            for id in &clears[root] {
                removable[id.0 as usize] = false;
            }
        }
    }
    // Writes through aliases also invalidate an input's original parameter
    // contents. Re-establish provenance below only from proven sources.
    parameter_storage.retain(|id| !incoming.contains_key(id));
    loop {
        let before = parameter_storage.len();
        for (&destination, sources) in &incoming {
            if sources
                .iter()
                .all(|source| parameter_storage.contains(source))
            {
                parameter_storage.insert(destination);
            }
        }
        if parameter_storage.len() == before {
            break;
        }
    }

    let mut needs_finite = vec![false; program.kernel_runs.len()];
    for run in program.kernel_calls() {
        if run
            .inputs
            .iter()
            .all(|input| clears[root(input.shard)].is_empty())
        {
            continue;
        }
        let call = run.call(target, None)?;
        for (operand, input) in run.inputs.iter().enumerate() {
            let readers = &clears[root(input.shard)];
            if readers.is_empty() {
                continue;
            }
            let (regions, finite) = match if operand == 0 {
                &call.padding
            } else {
                &PaddingRequirement::Required
            } {
                PaddingRequirement::Unread(regions) => (regions.as_slice(), false),
                PaddingRequirement::FiniteIfZero { region, zero }
                    if all_f16
                        && parameter_storage.contains(&root(run.inputs[1].shard))
                        && storage_location(&program.shards, run.inputs[1].shard).1 == 0
                        && program.shards[run.inputs[1].shard.index() as usize].extents
                            == program.shards[root(run.inputs[1].shard)].extents
                        && program.shards[run.inputs[1].shard.index() as usize]
                            .tensor_type
                            .format
                            .layout
                            .order
                            == program.shards[root(run.inputs[1].shard)]
                                .tensor_type
                                .format
                                .layout
                                .order
                        && zero
                            .iter()
                            .zip(&program.shards[run.inputs[1].shard.index() as usize].extents)
                            .any(|(region, storage)| region.start >= storage.logical_end) =>
                {
                    (std::slice::from_ref(region), true)
                }
                _ => (&[][..], false),
            };
            let ranges = regions
                .iter()
                .map(|region| {
                    let view = crate::ShardView {
                        shard: input.shard,
                        extents: region.clone(),
                    };
                    let bound = view.bind(&program.shards)?;
                    Ok((bound.traversal(CopyOrder::Physical)?, bound.backing.1))
                })
                .collect::<Result<Vec<_>, crate::storage::StorageError>>()?;
            for &id in readers {
                let clear = &program.kernel_runs[id.0 as usize];
                let MidOperationKind::FillZero { offset, bytes, .. } = clear.kernel else {
                    unreachable!();
                };
                let start =
                    storage_location(&program.shards, clear.outputs[0].shard).1 + i64::from(offset);
                let end = start + i64::from(bytes);
                removable[id.0 as usize] &= ranges.iter().any(|(ranges, base)| {
                    ranges.spans().any(|range| {
                        let begin = base + i64::from(range.offset);
                        begin <= start && end <= begin + i64::from(range.bytes)
                    })
                });
                needs_finite[id.0 as usize] |= finite;
            }
        }
    }
    let mut removed = 0;
    program.body.retain(&mut |operation| {
        if let BlockOperation::Compute { run: id, .. } = operation
            && removable[id.0 as usize]
        {
            program.requires_finite_scratch |= needs_finite[id.0 as usize];
            removed += 1;
            return false;
        }
        true
    });
    tracing::info!(
        removed,
        "eliminated padding clears using kernel access contracts"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{GemmKernelMode, GemmWeightLoad};
    use crate::mid::{MidInput, MidValueId};
    use crate::tensor::{AmpOrder, ElementOrder, Layout, ShardExtent, TensorTiling};
    use crate::{AccumulationPrecision, KernelRequirements, TensorType};

    fn fixture() -> TileGraph {
        let tensor_type = TensorType::new(
            [2, 48],
            Precision::F16,
            Layout::row_major(TensorTiling::replicated(1)),
        );
        let extents = vec![
            ShardExtent {
                axis: 0,
                start: 0,
                logical_end: 2,
                physical_end: 2,
            },
            ShardExtent {
                axis: 1,
                start: 0,
                logical_end: 48,
                physical_end: 64,
            },
        ];
        let view = |id| ShardView {
            shard: BlockValueId(id),
            extents: extents.clone(),
        };
        let run = |kernel, inputs: Vec<ShardView>, output| {
            let requirements = KernelRequirements {
                inputs: inputs.iter().map(|_| tensor_type.format.clone()).collect(),
                outputs: vec![tensor_type.format.clone()],
                distinct_elements: Vec::new(),
            };
            KernelRun::new(
                WorkProvenance {
                    operation: None,
                    value: None,
                    reason: WorkReason::LayoutRearrangement,
                },
                kernel,
                inputs,
                vec![view(output)],
                requirements,
            )
        };
        let mut program = TileGraph {
            tile_count: 1,
            requires_finite_scratch: false,
            shards: (0..3)
                .map(|id| BlockValue {
                    id: BlockValueId(id),
                    tile: 0,
                    tensor_type: tensor_type.clone(),
                    extents: extents.clone(),
                    definition: ShardDefinition::Staging,
                })
                .collect(),
            exchange_phases: Vec::new(),
            inputs: vec![MidInput {
                name: "weights".into(),
                kind: crate::GraphInputKind::Parameter,
                value: MidValueId::from_index(0),
            }],
            body: BlockRegion {
                operations: vec![
                    BlockOperation::Compute {
                        tile: 0,
                        run: KernelRunId(0),
                    },
                    BlockOperation::Compute {
                        tile: 0,
                        run: KernelRunId(1),
                    },
                ],
            },
            kernel_runs: vec![
                run(
                    MidOperationKind::FillZero {
                        offset: 96,
                        bytes: 32,
                        padding_only: true,
                    },
                    Vec::new(),
                    0,
                ),
                run(
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
                    vec![view(0), view(1)],
                    2,
                ),
            ],
            local_copies: Vec::new(),
            value_views: vec![vec![view(1)], vec![view(0)]],
            outputs: Vec::new(),
            logical_values: Vec::new(),
            checkpoints: Vec::new(),
        };
        for (id, rows, columns) in [(1, 48, 16), (2, 2, 16)] {
            let shard = &mut program.shards[id];
            shard.tensor_type.shape = crate::TensorShape(vec![rows, columns]);
            shard.extents[0].logical_end = rows;
            shard.extents[0].physical_end = if id == 1 { 64 } else { rows };
            shard.extents[1].logical_end = columns;
            shard.extents[1].physical_end = columns;
        }
        program.kernel_runs[1].inputs[1].extents = program.shards[1].extents.clone();
        program.kernel_runs[1].outputs[0].extents = program.shards[2].extents.clone();
        program.shards[2].tensor_type.format.layout.memory_class =
            crate::MemoryClass::Ipu21Interleaved;
        Arc::make_mut(&mut program.kernel_runs[1].metadata)
            .requirements
            .outputs[0] = program.shards[2].tensor_type.format.clone();
        program
    }

    fn has_clear(program: &TileGraph) -> bool {
        program
            .kernel_calls()
            .any(|run| matches!(run.kernel, MidOperationKind::FillZero { .. }))
    }

    #[test]
    fn padding_removal_changes_the_graph_cost_and_projected_repeat_work_together() {
        for count in [1, 3] {
            let mut program = fixture();
            if count > 1 {
                let body = std::mem::take(&mut program.body);
                program
                    .body
                    .operations
                    .push(BlockOperation::Repeat(Box::new(BlockRepeat {
                        provenance: program.kernel_runs[1].provenance,
                        count,
                        bindings: vec![BlockRepeatBinding {
                            tile: 0,
                            carried: vec![],
                            invariants: vec![],
                            iterated: vec![],
                        }],
                        body,
                    })));
            }
            let before =
                crate::estimate::scheduled_program_cycles(Target::Ipu21, &program, &[]).unwrap();
            eliminate(Target::Ipu21, &mut program).unwrap();
            let after =
                crate::estimate::scheduled_program_cycles(Target::Ipu21, &program, &[]).unwrap();
            assert!(after.total < before.total);
            assert!(!has_clear(&program));
            assert!(program.requires_finite_scratch);
            // The removed entry may remain interned, but must not become live
            // again when a consumer derives per-tile execution indexes.
            assert_eq!(program.kernel_runs.len(), 2);
            let program = Arc::new(program);
            let low = lower_to_tiles(&program, false);
            assert!(Arc::ptr_eq(&program, &low.program));
            let work = if count == 1 {
                &low.tiles[0]
            } else {
                &low.repeat_runs[0].body
            };
            let [BlockOperation::Compute { run: id, .. }] = work.work.as_slice() else {
                panic!("projection did not retain exactly the live computation");
            };
            let mut executed = (*program).clone();
            executed.body.operations = vec![BlockOperation::Compute { tile: 0, run: *id }];
            let cost =
                crate::estimate::scheduled_program_cycles(Target::Ipu21, &executed, &[]).unwrap();
            assert_eq!(after.total, cost.total * u64::from(count));
            let mut again = (*program).clone();
            eliminate(Target::Ipu21, &mut again).unwrap();
            assert_eq!(again, *program, "removal must be idempotent");
        }
    }

    fn append_copy(program: &mut TileGraph, source: u32, destination: u32, bytes: u32) {
        let copy = LocalCopyId(program.local_copies.len() as u32);
        program.local_copies.push(
            crate::kernel::CopyRun::bind(
                LocalCopy {
                    source: BlockValueId(source),
                    source_offset: 0,
                    destination: BlockValueId(destination),
                    destination_offset: 0,
                    bytes,
                    pattern: CopyPattern::Contiguous,
                },
                &program.shards,
            )
            .unwrap(),
        );
        program
            .body
            .operations
            .insert(0, BlockOperation::Copy { tile: 0, copy });
    }

    #[test]
    fn cast_padding_elision_requires_exclusive_column_only_reader() {
        let mut baseline = fixture();
        let graph = &mut baseline;
        graph.shards[2] = BlockValue {
            id: BlockValueId(2),
            ..graph.shards[0].clone()
        };
        graph.shards[0].extents[1].logical_end = 48;
        let run = &mut graph.kernel_runs[1];
        run.inputs.truncate(1);
        run.outputs[0].extents = graph.shards[2].extents.clone();
        run.inputs[0].extents[1].logical_end = 48;
        let metadata = Arc::make_mut(&mut run.metadata);
        metadata.requirements.inputs.truncate(1);
        metadata.kernel = MidOperationKind::Cast {
            from: Precision::F16,
            to: Precision::F8F143 { scale_exponent: -4 },
        };
        metadata.requirements.outputs[0].layout.order = ElementOrder::Amp(AmpOrder::Left);
        metadata.requirements.outputs[0].precision = Precision::F8F143 { scale_exponent: -4 };
        graph.shards[2].tensor_type.format = metadata.requirements.outputs[0].clone();
        for case in 0..15 {
            let mut program = baseline.clone();
            let graph = &mut program;
            match case {
                1 => {
                    graph.shards[0].extents[0].logical_end = 1;
                    graph.kernel_runs[1].inputs[0].extents[0].logical_end = 1;
                }
                2 => graph.outputs.push(MidValueId::from_index(1)),
                3 => append_copy(graph, 0, 1, 128),
                5 | 6 => {
                    // Non-vector-aligned logical columns select physical reads.
                    graph.shards[0].extents[1].logical_end = 50;
                    graph.kernel_runs[1].inputs[0].extents[1].logical_end = 50;
                    if case == 6 {
                        graph.shards[0].extents[0].logical_end = 1;
                        graph.kernel_runs[1].inputs[0].extents[0].logical_end = 1;
                    }
                }
                4 => {
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        MidOperationKind::FillZero {
                            offset: 96,
                            bytes: 32,
                            padding_only: false,
                        };
                }
                7..=10 => {
                    let mut argument = graph.shards[1].clone();
                    argument.id = BlockValueId(3);
                    graph.shards.push(argument);
                    let body = graph.body.clone();
                    graph
                        .body
                        .operations
                        .push(BlockOperation::Repeat(Box::new(BlockRepeat {
                            provenance: graph.kernel_runs[1].provenance,
                            count: 3,
                            bindings: vec![BlockRepeatBinding {
                                tile: 0,
                                carried: Vec::new(),
                                invariants: if case == 10 {
                                    vec![RepeatInvariant {
                                        input: BlockValueId(1),
                                        argument: BlockValueId(0),
                                    }]
                                } else {
                                    Vec::new()
                                },
                                iterated: vec![RepeatIterated {
                                    inputs: vec![BlockValueId(if case == 9 { 0 } else { 1 }); 3],
                                    argument: BlockValueId(if case == 8 { 0 } else { 3 }),
                                }],
                            }],
                            body,
                        })));
                }
                11 | 12 => {
                    let physical = if case == 11 { 2 } else { 65536 };
                    let logical = physical - 1;
                    for id in [0, 2] {
                        graph.shards[id].extents[0].logical_end = logical;
                        graph.shards[id].extents[0].physical_end = physical;
                    }
                    graph.kernel_runs[1].inputs[0].extents = graph.shards[0].extents.clone();
                    graph.kernel_runs[1].outputs[0].extents = graph.shards[2].extents.clone();
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        MidOperationKind::FillZero {
                            offset: logical * 128,
                            bytes: 128,
                            padding_only: true,
                        };
                }
                13 => {
                    // A clear crossing from live data into padding cannot be dropped.
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        MidOperationKind::FillZero {
                            offset: 88,
                            bytes: 40,
                            padding_only: true,
                        };
                }
                14 => {
                    // Another alias reads the first four elements of the padding.
                    let mut alias = graph.shards[0].clone();
                    alias.id = BlockValueId(3);
                    alias.definition = ShardDefinition::Alias(BlockValueId(0));
                    alias.extents[1].logical_end = 52;
                    let mut reader = graph.kernel_runs[1].clone();
                    reader.inputs[0] = ShardView {
                        shard: alias.id,
                        extents: alias.extents.clone(),
                    };
                    graph.shards.push(alias);
                    graph.kernel_runs.push(reader);
                    graph.body.operations.push(BlockOperation::Compute {
                        tile: 0,
                        run: KernelRunId(2),
                    });
                }
                _ => {}
            }
            eliminate(Target::Ipu21, &mut program).unwrap();
            assert_eq!(
                has_clear(&program),
                !(case <= 1 || case == 7 || case == 11),
                "case {case}"
            );
        }
    }

    #[test]
    fn finite_padding_requires_zero_weights_and_no_other_consumers() {
        let mut program = fixture();
        eliminate(Target::Ipu21, &mut program).unwrap();
        assert!(!has_clear(&program));
        assert!(program.requires_finite_scratch);
        for case in 0..8 {
            let mut program = fixture();
            let graph = &mut program;
            match case {
                0 => graph.inputs.clear(),
                1 => graph.shards[2].tensor_type.format.precision = Precision::F32,
                2 => {
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        MidOperationKind::FillZero {
                            offset: 96,
                            bytes: 32,
                            padding_only: false,
                        }
                }
                3 => {
                    graph.shards[0].extents[0].logical_end = 1;
                    graph.kernel_runs[1].inputs[0].extents[0].logical_end = 1;
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        MidOperationKind::FillZero {
                            offset: 224,
                            bytes: 32,
                            padding_only: true,
                        };
                }
                4 => graph.outputs.push(MidValueId::from_index(1)),
                5 => {
                    graph.shards[1].extents[0].logical_end = 56;
                    graph.kernel_runs[1].inputs[1].extents[0].logical_end = 56;
                }
                6 => {
                    // A parameter alias with a shifted K origin is not a matching coefficient panel.
                    graph.shards[1].extents[0].start = 8;
                    graph.kernel_runs[1].inputs[1].extents[0].start = 8;
                }
                _ => {
                    // A narrower call view does not make live parameter coefficients zero.
                    graph.shards[1].extents[0].logical_end = 64;
                }
            }
            eliminate(Target::Ipu21, &mut program).unwrap();
            assert!(has_clear(&program), "case {case}");
        }
    }

    #[test]
    fn parameter_copy_proof_rejects_mixed_sources() {
        let mut program = fixture();
        let graph = &mut program;
        let mut staged = graph.shards[1].clone();
        staged.id = BlockValueId(3);
        graph.shards.push(staged);
        append_copy(graph, 1, 3, 2048);
        graph.kernel_runs[1].inputs[1].shard = BlockValueId(3);
        let mut mixed = program.clone();
        append_copy(&mut mixed, 2, 3, 8);
        let mut overwritten = mixed.clone();
        let last = overwritten.local_copies.last_mut().unwrap();
        let mut movement = last.movement().clone();
        movement.destination = BlockValueId(1);
        *last = crate::kernel::CopyRun::bind(movement, &overwritten.shards).unwrap();
        eliminate(Target::Ipu21, &mut program).unwrap();
        eliminate(Target::Ipu21, &mut mixed).unwrap();
        eliminate(Target::Ipu21, &mut overwritten).unwrap();
        assert!(!has_clear(&program));
        assert!(has_clear(&mixed));
        assert!(has_clear(&overwritten));
    }

    #[test]
    fn parameter_copy_must_preserve_the_zero_region() {
        for partial in [false, true] {
            let mut program = fixture();
            let mut staged = program.shards[1].clone();
            staged.id = BlockValueId(3);
            program.shards.push(staged);
            if !partial {
                // Live coefficients are copied into the narrower destination's padding.
                program.shards[1].extents[0].logical_end = 64;
                program.shards[1].tensor_type.shape.0[0] = 64;
                program.value_views[0][0].extents = program.shards[1].extents.clone();
            }
            append_copy(&mut program, 1, 3, if partial { 256 } else { 2048 });
            program.kernel_runs[1].inputs[1].shard = BlockValueId(3);
            eliminate(Target::Ipu21, &mut program).unwrap();
            assert!(has_clear(&program), "partial={partial}");
        }
    }

    #[test]
    fn parameter_exchange_must_cover_identical_padding() {
        for case in 0..3 {
            let mut program = fixture();
            let mut staged = program.shards[1].clone();
            staged.id = BlockValueId(3);
            program.shards.push(staged);
            if case == 1 {
                program.shards[1].extents[0].logical_end = 64;
                program.shards[1].tensor_type.shape.0[0] = 64;
                program.value_views[0][0].extents = program.shards[1].extents.clone();
            }
            let mut source = ShardView {
                shard: BlockValueId(1),
                extents: program.shards[1].extents.clone(),
            };
            let mut destination = ShardView {
                shard: BlockValueId(3),
                extents: program.shards[3].extents.clone(),
            };
            if case == 2 {
                source.extents[0].physical_end = 48;
                destination.extents[0].physical_end = 48;
            }
            program.exchange_phases.push(ExchangePhase {
                id: ExchangePhaseId(0),
                provenance: program.kernel_runs[1].provenance,
                transfers: vec![LogicalExchange {
                    source,
                    destinations: vec![destination],
                    order: CopyOrder::Physical,
                }],
            });
            program
                .body
                .operations
                .insert(0, BlockOperation::Exchange(ExchangePhaseId(0)));
            program.kernel_runs[1].inputs[1].shard = BlockValueId(3);
            eliminate(Target::Ipu21, &mut program).unwrap();
            assert_eq!(has_clear(&program), case != 0, "case={case}");
        }
    }
}
