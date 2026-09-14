//! Remove padding initialization when readers do not need it.
//!
//! Packing kernels can skip unread input padding and write their own zeros.
//! Separately, numerical inputs and operation results are assumed finite. For an all-F16
//! program, zeroing the tensor arena once establishes an invariant preserved
//! by writes, copies and allocation reuse. This second proof cannot apply to
//! mixed-precision arenas: finite FP32 bits need not encode finite FP16 values.

use super::*;
use crate::kernel::TileKernelSpec;
use crate::tensor::{AmpOrder, ElementOrder, Precision};
use std::collections::BTreeSet;

// These readers have no kernel-specific proof that padding can be ignored.
// Repeat bindings can expose bytes to readers under another shard ID; scratch
// local to the body has no such additional readers.
fn non_kernel_read_storage(program: &TileGraph) -> BTreeSet<BlockValueId> {
    let mut readers = program
        .outputs
        .iter()
        .flat_map(|&output| program.value_shards(output).iter().copied())
        .collect::<BTreeSet<_>>();
    for operation in program.body.walk() {
        match operation {
            BlockOperation::Exchange(id) => readers.extend(
                program.exchange_phases[id.index() as usize]
                    .transfers
                    .iter()
                    .map(|transfer| transfer.source.shard),
            ),
            BlockOperation::Copy { copy, .. } => {
                readers.insert(program.local_copies[copy.0 as usize].source);
            }
            BlockOperation::Repeat(repeat) => {
                readers.extend(
                    repeat
                        .bindings
                        .iter()
                        .flat_map(BlockRepeatBinding::bound_shards),
                );
            }
            _ => {}
        }
    }
    readers
        .into_iter()
        .map(|id| storage_root(&program.shards, id))
        .collect()
}

/// Row-major FP8 packing reads only logical columns/rows and writes its own
/// output padding. Drop input padding clears when that is the sole reader.
pub(super) fn omit_unread_fp8_input_padding(program: &mut TileGraph) {
    let shards = &program.shards;
    let kernels = &program.kernel_runs;
    let root = |id| storage_root(shards, id);
    let mut candidates = BTreeSet::new();
    let mut forbidden = non_kernel_read_storage(program);
    for run in program.kernel_calls() {
        for input in &run.inputs {
            for view in &input.views {
                let block = &shards[view.shard.index() as usize];
                let columns = view.extents.last();
                let ignores_padding = matches!(run.kernel, TileKernelSpec::Cast {
                    from: Precision::F16, to: Precision::F8F143 { .. }
                } | TileKernelSpec::Gelu)
                    && matches!(run.requirements.outputs[0].format.precision, Precision::F8F143 { .. })
                    && block.tensor_type.format.precision == Precision::F16
                    && block.tensor_type.format.layout.order == ElementOrder::RowMajor
                    && matches!(run.requirements.outputs[0].format.layout.order, ElementOrder::Amp(AmpOrder::Left) | ElementOrder::RowMajor)
                    && (run.kernel == TileKernelSpec::Gelu || run.requirements.outputs[0].format.layout.order == ElementOrder::Amp(AmpOrder::Left))
                    && view.extents == block.extents
                    && columns.is_some_and(|axis| (axis.logical_end - axis.start).is_multiple_of(4))
                    // Matrix-row padding is skipped by the packed row bounds;
                    // padding in outer dimensions still requires initialization.
                    && view.extents.iter().rev().skip(2).all(|axis| axis.logical_end == axis.physical_end)
                    && (view.extents.len() < 2 || {
                        let rows = view.extents[view.extents.len()-2];
                        rows.logical_end == rows.physical_end || (matches!(run.kernel, TileKernelSpec::Cast { .. }) && rows.physical_end - rows.start <= u16::MAX.into())
                    });
                if ignores_padding {
                    candidates.insert(root(view.shard));
                } else {
                    forbidden.insert(root(view.shard));
                }
            }
        }
        if !matches!(run.kernel, TileKernelSpec::FillZero { .. }) {
            forbidden.extend(run.outputs.iter().map(|view| root(view.shard)));
        }
    }
    candidates.retain(|id| !forbidden.contains(id));
    let mut removed = 0;
    let mut keep = |operation: &BlockOperation| {
        if let BlockOperation::Compute { run: id, .. } = operation {
            let run = &kernels[id.0 as usize];
            if matches!(
                run.kernel,
                TileKernelSpec::FillZero {
                    padding_only: true,
                    ..
                }
            ) && candidates.contains(&root(run.outputs[0].shard))
            {
                removed += 1;
                return false;
            }
        }
        true
    };
    program.body.retain(&mut keep);
    tracing::info!(removed, "eliminated unread FP8 cast input padding clears");
}

pub(super) fn reuse_finite_padding(program: &mut TileGraph) {
    if program
        .shards
        .iter()
        .any(|shard| shard.tensor_type.format.precision != Precision::F16)
        // Attention row-state storage contains FP32 words inside its packed
        // F16 allocation. Its bits do not preserve the arena-wide invariant.
        || program.kernel_calls().any(|run| matches!(run.kernel,
            TileKernelSpec::AttentionSoftmax { .. } | TileKernelSpec::AttentionMerge { .. }
        ))
    {
        return;
    }
    let root = |id| storage_root(&program.shards, id);
    let mut parameter_storage = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        .flat_map(|input| program.value_shards(input.value).iter().copied())
        .map(root)
        .collect::<BTreeSet<_>>();
    let mut incoming = std::collections::BTreeMap::<BlockValueId, BTreeSet<BlockValueId>>::new();
    for operation in program.body.walk() {
        match operation {
            BlockOperation::Copy { copy, .. } => {
                let copy = &program.local_copies[copy.0 as usize];
                incoming
                    .entry(root(copy.destination))
                    .or_default()
                    .insert(root(copy.source));
            }
            BlockOperation::Exchange(id) => {
                for transfer in &program.exchange_phases[id.index() as usize].transfers {
                    for destination in &transfer.destinations {
                        incoming
                            .entry(root(destination.shard))
                            .or_default()
                            .insert(root(transfer.source.shard));
                    }
                }
            }
            BlockOperation::Compute { run, .. } => {
                let run = &program.kernel_runs[run.0 as usize];
                if !matches!(run.kernel, TileKernelSpec::FillZero { .. }) {
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
    let mut candidates = BTreeSet::new();
    let mut forbidden = non_kernel_read_storage(program);
    for run in program.kernel_calls() {
        // Parameter packing supplies exact zero coefficients beyond logical K.
        // Only the activation operand can therefore tolerate arbitrary finite K
        // padding. Missing logical values and explicit reduction zeros cannot.
        let finite_left = matches!(run.kernel, TileKernelSpec::Gemm { .. })
            && run.inputs.len() == 2
            && !run.inputs[1].views.is_empty()
            && run.inputs[1]
                .views
                .iter()
                .all(|view| parameter_storage.contains(&root(view.shard)));
        for (index, input) in run.inputs.iter().enumerate() {
            for view in &input.views {
                if finite_left && index == 0 {
                    candidates.insert(root(view.shard));
                } else {
                    forbidden.insert(root(view.shard));
                }
            }
        }
        if !matches!(run.kernel, TileKernelSpec::FillZero { .. }) {
            forbidden.extend(run.outputs.iter().map(|output| root(output.shard)));
        }
    }
    candidates.retain(|shard| !forbidden.contains(shard));
    let shards = &program.shards;
    let kernels = &program.kernel_runs;
    let mut valid_row_ranges = std::collections::BTreeMap::new();
    let mut removed = 0;
    let mut keep = |operation: &BlockOperation| {
        if let BlockOperation::Compute { run: id, .. } = operation {
            let run = &kernels[id.0 as usize];
            if let TileKernelSpec::FillZero {
                offset,
                bytes,
                padding_only: true,
            } = run.kernel
                && candidates.contains(&storage_root(shards, run.outputs[0].shard))
            {
                // Keep discarded row padding zero: arbitrary nonzero rows
                // could overflow even though their outputs are unobserved.
                let ranges = valid_row_ranges
                    .entry(run.outputs[0].shard)
                    .or_insert_with(|| {
                        let shard = &shards[run.outputs[0].shard.index() as usize];
                        let mut rows = shard.extents.clone();
                        let rank = rows.len();
                        let inner = match shard.tensor_type.format.layout.order {
                            ElementOrder::Amp(AmpOrder::Left) | ElementOrder::RowMajor => {
                                rank.checked_sub(1)?
                            }
                            ElementOrder::Amp(AmpOrder::TransposedLeft) => rank.checked_sub(2)?,
                            _ => return None,
                        };
                        for (axis, extent) in rows.iter_mut().enumerate() {
                            if axis != inner {
                                extent.physical_end = extent.logical_end;
                            }
                        }
                        crate::storage::byte_traversal(shard.storage(), &rows, true).ok()
                    });
                if !ranges.as_ref().is_some_and(|ranges| {
                    ranges.spans().any(|range| {
                        range.offset <= offset
                            && u64::from(offset) + u64::from(bytes)
                                <= u64::from(range.offset) + u64::from(range.bytes)
                    })
                }) {
                    return true;
                }
                removed += 1;
                return false;
            }
        }
        true
    };
    program.body.retain(&mut keep);
    program.requires_finite_scratch |= removed != 0;
    tracing::info!(
        removed,
        "eliminated finite GEMM padding clears using load-time SRAM initialization"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{GemmKernelMode, GemmWeightLoad};
    use crate::mid::{MidInput, MidValueId};
    use crate::tensor::{Layout, ShardExtent, TensorTiling};
    use crate::{AccumulationPrecision, KernelAccess, KernelRequirements, TensorType};

    fn fixture() -> TileGraph {
        let tensor_type = TensorType::new(
            [2, 64],
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
                logical_end: 50,
                physical_end: 64,
            },
        ];
        let view = |id| ShardView {
            shard: BlockValueId(id),
            extents: extents.clone(),
        };
        let run = |kernel, inputs, output| {
            KernelRun::new(
                WorkProvenance {
                    operation: None,
                    value: None,
                    reason: WorkReason::LayoutRearrangement,
                },
                kernel,
                inputs,
                vec![view(output)],
                KernelRequirements {
                    inputs: Vec::new(),
                    outputs: vec![KernelAccess::new(tensor_type.format.clone(), 8)],
                    distinct_elements: Vec::new(),
                },
            )
        };
        TileGraph {
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
                    TileKernelSpec::FillZero {
                        offset: 96,
                        bytes: 32,
                        padding_only: true,
                    },
                    Vec::new(),
                    0,
                ),
                run(
                    TileKernelSpec::Gemm {
                        multiply: Precision::F16,
                        accumulate: AccumulationPrecision::F32,
                        mode: GemmKernelMode::Initialize,
                        weights: GemmWeightLoad::Standard,
                        inner_block: 64,
                        output_columns: 16,
                    },
                    vec![
                        KernelOperand {
                            views: vec![view(0)],
                        },
                        KernelOperand {
                            views: vec![view(1)],
                        },
                    ],
                    2,
                ),
            ],
            local_copies: Vec::new(),
            value_shards: vec![vec![BlockValueId(1)], vec![BlockValueId(0)]],
            outputs: Vec::new(),
            logical_values: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    fn has_clear(program: &TileGraph) -> bool {
        program
            .kernel_calls()
            .any(|run| matches!(run.kernel, TileKernelSpec::FillZero { .. }))
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
            let before = crate::estimate::scheduled_program_cycles(&program, &[]).unwrap();
            reuse_finite_padding(&mut program);
            let after = crate::estimate::scheduled_program_cycles(&program, &[]).unwrap();
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
            let [TileWork::Kernel(id)] = work.work.as_slice() else {
                panic!("projection did not retain exactly the live computation");
            };
            let mut executed = (*program).clone();
            executed.body.operations = vec![BlockOperation::Compute { tile: 0, run: *id }];
            let cost = crate::estimate::scheduled_program_cycles(&executed, &[]).unwrap();
            assert_eq!(after.total, cost.total * u64::from(count));
            let mut again = (*program).clone();
            reuse_finite_padding(&mut again);
            assert_eq!(again, *program, "removal must be idempotent");
        }
    }

    fn append_copy(program: &mut TileGraph, source: u32, destination: u32, bytes: u32) {
        let copy = LocalCopyId(program.local_copies.len() as u32);
        program.local_copies.push(LocalCopy {
            source: BlockValueId(source),
            source_offset: 0,
            destination: BlockValueId(destination),
            destination_offset: 0,
            bytes,
            pattern: CopyPattern::Contiguous,
        });
        program
            .body
            .operations
            .insert(0, BlockOperation::Copy { tile: 0, copy });
    }

    #[test]
    fn cast_padding_elision_requires_exclusive_column_only_reader() {
        let mut baseline = fixture();
        let graph = &mut baseline;
        graph.shards[0].extents[1].logical_end = 48;
        let run = &mut graph.kernel_runs[1];
        run.inputs.truncate(1);
        run.inputs[0].views[0].extents[1].logical_end = 48;
        let metadata = Arc::make_mut(&mut run.metadata);
        metadata.kernel = TileKernelSpec::Cast {
            from: Precision::F16,
            to: Precision::F8F143 { scale_exponent: -4 },
        };
        metadata.requirements.outputs[0].format.layout.order = ElementOrder::Amp(AmpOrder::Left);
        metadata.requirements.outputs[0].format.precision =
            Precision::F8F143 { scale_exponent: -4 };
        for case in 0..11 {
            let mut program = baseline.clone();
            let graph = &mut program;
            match case {
                1 => {
                    graph.shards[0].extents[0].logical_end = 1;
                    graph.kernel_runs[1].inputs[0].views[0].extents[0].logical_end = 1;
                }
                2 => graph.outputs.push(MidValueId::from_index(1)),
                3 => append_copy(graph, 0, 1, 128),
                5 | 6 => {
                    Arc::make_mut(&mut graph.kernel_runs[1].metadata).kernel = TileKernelSpec::Gelu;
                    if case == 6 {
                        graph.shards[0].extents[0].logical_end = 1;
                        graph.kernel_runs[1].inputs[0].views[0].extents[0].logical_end = 1;
                    }
                }
                4 => {
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        TileKernelSpec::FillZero {
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
                _ => {}
            }
            omit_unread_fp8_input_padding(&mut program);
            assert_eq!(
                has_clear(&program),
                !(case <= 1 || case == 5 || case == 7),
                "case {case}"
            );
        }
    }

    #[test]
    fn embedded_fp32_attention_state_prevents_f16_arena_reuse() {
        let mut program = fixture();
        let graph = &mut program;
        let mut state_writer = graph.kernel_runs[1].clone();
        state_writer.inputs.clear();
        Arc::make_mut(&mut state_writer.metadata).kernel = TileKernelSpec::AttentionSoftmax {
            head_dimension: 72,
            key_columns: 729,
            padded_key_columns: 768,
        };
        let run = KernelRunId(graph.kernel_runs.len() as u32);
        graph.kernel_runs.push(state_writer);
        graph
            .body
            .operations
            .push(BlockOperation::Compute { tile: 0, run });
        reuse_finite_padding(&mut program);
        assert!(has_clear(&program));
    }

    #[test]
    fn finite_padding_requires_zero_weights_and_no_other_consumers() {
        let mut program = fixture();
        reuse_finite_padding(&mut program);
        assert!(!has_clear(&program));
        assert!(program.requires_finite_scratch);
        for case in 0..5 {
            let mut program = fixture();
            let graph = &mut program;
            match case {
                0 => graph.inputs.clear(),
                1 => graph.shards[2].tensor_type.format.precision = Precision::F32,
                2 => {
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        TileKernelSpec::FillZero {
                            offset: 96,
                            bytes: 32,
                            padding_only: false,
                        }
                }
                3 => {
                    graph.shards[0].extents[0].logical_end = 1;
                    Arc::make_mut(&mut graph.kernel_runs[0].metadata).kernel =
                        TileKernelSpec::FillZero {
                            offset: 224,
                            bytes: 32,
                            padding_only: true,
                        };
                }
                _ => graph.outputs.push(MidValueId::from_index(1)),
            }
            reuse_finite_padding(&mut program);
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
        append_copy(graph, 1, 3, 256);
        graph.kernel_runs[1].inputs[1].views[0].shard = BlockValueId(3);
        let mut mixed = program.clone();
        append_copy(&mut mixed, 2, 3, 8);
        let mut overwritten = mixed.clone();
        overwritten.local_copies.last_mut().unwrap().destination = BlockValueId(1);
        reuse_finite_padding(&mut program);
        reuse_finite_padding(&mut mixed);
        reuse_finite_padding(&mut overwritten);
        assert!(!has_clear(&program));
        assert!(has_clear(&mixed));
        assert!(has_clear(&overwritten));
    }
}
