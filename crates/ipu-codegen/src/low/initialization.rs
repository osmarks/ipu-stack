//! Reuse finite SRAM contents for GEMM activation K padding.
//!
//! Numerical inputs and operation results are assumed finite. For an all-F16
//! program, zeroing the tensor arena once establishes an invariant preserved
//! by writes, copies and allocation reuse. Mixed-precision arenas deliberately
//! retain their clears: finite FP32 bits need not encode finite FP16 values.

use super::*;
use std::collections::BTreeSet;

pub(super) fn reuse_finite_padding(program: &mut LowProgram) {
    if program
        .shards
        .iter()
        .any(|shard| shard.tensor_type.format.precision != Precision::F16)
        // Attention row-state storage contains FP32 words inside its packed
        // F16 allocation. Its bits do not preserve the arena-wide invariant.
        || program.kernel_runs.iter().any(|run| matches!(run.kernel,
            TileKernelSpec::AttentionSoftmax { .. } | TileKernelSpec::AttentionMerge { .. }
        ))
    {
        return;
    }
    let parameters = program
        .inputs
        .iter()
        .filter(|input| input.kind == crate::GraphInputKind::Parameter)
        .flat_map(|input| input.shards.iter().copied())
        .collect::<BTreeSet<_>>();
    let root = |id| storage_root(&program.shards, id);
    let mut parameter_storage = parameters
        .iter()
        .map(|&id| root(id))
        .collect::<BTreeSet<_>>();
    let mut incoming = std::collections::BTreeMap::<BlockValueId, BTreeSet<BlockValueId>>::new();
    for copy in &program.local_copies {
        incoming
            .entry(root(copy.destination))
            .or_default()
            .insert(root(copy.source));
    }
    for phase in &program.exchange_phases {
        for transfer in &phase.transfers {
            for destination in &transfer.destinations {
                incoming
                    .entry(root(destination.shard))
                    .or_default()
                    .insert(root(transfer.source.shard));
            }
        }
    }
    for run in &program.kernel_runs {
        if !matches!(run.kernel, TileKernelSpec::FillZero { .. }) {
            // Arithmetic results are not known-zero-padded parameters.
            parameter_storage.remove(&root(run.output.shard));
            incoming
                .entry(root(run.output.shard))
                .or_default()
                .insert(root(run.output.shard));
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
    let mut forbidden = BTreeSet::new();
    for run in &program.kernel_runs {
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
            forbidden.insert(root(run.output.shard));
        }
    }
    for phase in &program.exchange_phases {
        for exchange in &phase.transfers {
            forbidden.insert(root(exchange.source.shard));
        }
    }
    for copy in &program.local_copies {
        forbidden.insert(root(copy.source));
    }
    for output in &program.outputs {
        forbidden.extend(output.shards.iter().map(|&id| root(id)));
    }
    // A carried/iterated binding gives the same bytes another semantic role.
    for repeat in &program.repeat_runs {
        for binding in &repeat.carried {
            forbidden.extend(
                [
                    binding.initial,
                    binding.argument,
                    binding.yielded,
                    binding.result,
                ]
                .map(root),
            );
        }
        if !repeat.iterated.is_empty() || !repeat.invariants.is_empty() {
            return;
        }
    }
    candidates.retain(|shard| !forbidden.contains(shard));
    let graph = &program.program;
    let kernels = &graph.kernel_runs;
    let mut valid_row_ranges = std::collections::BTreeMap::new();
    let mut removed = 0;
    let mut remove = |work: &TileWork| {
        if let TileWork::Kernel(id) = work {
            let run = &kernels[id.0 as usize];
            if let TileKernelSpec::FillZero {
                offset,
                bytes,
                padding_only: true,
            } = run.kernel
                && candidates.contains(&storage_root(&graph.shards, run.output.shard))
            {
                // Keep discarded row padding zero: arbitrary nonzero rows
                // could overflow even though their outputs are unobserved.
                let ranges = valid_row_ranges.entry(run.output.shard).or_insert_with(|| {
                    let shard = &graph.shards[run.output.shard.index() as usize];
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
                    crate::storage::physical_byte_spans(shard.storage(), &rows).ok()
                });
                if !ranges.as_ref().is_some_and(|ranges| {
                    ranges.iter().any(|range| {
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
    for tile in &mut program.tiles {
        tile.work.retain(&mut remove);
    }
    for repeat in &mut program.repeat_runs {
        repeat.body.work.retain(&mut remove);
    }
    tracing::info!(
        removed,
        "eliminated finite GEMM padding clears using load-time SRAM initialization"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccumulationPrecision, KernelAccess, KernelRequirements, TensorType};

    fn fixture() -> LowProgram {
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
                view(output),
                KernelRequirements {
                    inputs: Vec::new(),
                    output: KernelAccess::new(tensor_type.format.clone(), 8),
                    distinct_elements: Vec::new(),
                },
            )
        };
        let graph = TileGraph {
            tile_count: 1,
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
            inputs: vec![ProgramInput {
                name: "weights".into(),
                kind: crate::GraphInputKind::Parameter,
                value: MidValueId::from_index(0),
                shards: vec![BlockValueId(1)],
            }],
            body: BlockRegion::default(),
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
            values: Vec::new(),
            outputs: Vec::new(),
            logical_values: Vec::new(),
            checkpoints: Vec::new(),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        };
        LowProgram {
            program: Arc::new(graph),
            tiles: vec![TileWorkList {
                tile: 0,
                work: vec![
                    TileWork::Kernel(KernelRunId(0)),
                    TileWork::Kernel(KernelRunId(1)),
                ],
            }],
            repeat_runs: Vec::new(),
        }
    }

    #[test]
    fn embedded_fp32_attention_state_prevents_f16_arena_reuse() {
        let mut program = fixture();
        let graph = Arc::make_mut(&mut program.program);
        let mut state_writer = graph.kernel_runs[1].clone();
        state_writer.inputs.clear();
        Arc::make_mut(&mut state_writer.metadata).kernel = TileKernelSpec::AttentionSoftmax {
            head_dimension: 72,
            key_columns: 729,
            padded_key_columns: 768,
        };
        graph.kernel_runs.push(state_writer);
        reuse_finite_padding(&mut program);
        assert_eq!(program.tiles[0].work.len(), 2);
    }

    #[test]
    fn finite_padding_requires_zero_weights_and_no_other_consumers() {
        let mut program = fixture();
        reuse_finite_padding(&mut program);
        assert_eq!(program.tiles[0].work.len(), 1);
        for case in 0..5 {
            let mut program = fixture();
            let graph = Arc::make_mut(&mut program.program);
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
                _ => graph.outputs.push(ValueBlocks {
                    value: MidValueId::from_index(0),
                    shards: vec![BlockValueId(0)],
                }),
            }
            reuse_finite_padding(&mut program);
            assert_eq!(program.tiles[0].work.len(), 2, "case {case}");
        }
    }

    #[test]
    fn parameter_copy_proof_rejects_mixed_sources() {
        let mut program = fixture();
        let graph = Arc::make_mut(&mut program.program);
        let mut staged = graph.shards[1].clone();
        staged.id = BlockValueId(3);
        graph.shards.push(staged);
        graph.local_copies.push(LocalCopy {
            source: BlockValueId(1),
            source_offset: 0,
            destination: BlockValueId(3),
            destination_offset: 0,
            bytes: 256,
            pattern: CopyPattern::Contiguous,
        });
        graph.kernel_runs[1].inputs[1].views[0].shard = BlockValueId(3);
        let mut mixed = program.clone();
        Arc::make_mut(&mut mixed.program)
            .local_copies
            .push(LocalCopy {
                source: BlockValueId(2),
                source_offset: 0,
                destination: BlockValueId(3),
                destination_offset: 0,
                bytes: 8,
                pattern: CopyPattern::Contiguous,
            });
        let mut overwritten = mixed.clone();
        Arc::make_mut(&mut overwritten.program)
            .local_copies
            .last_mut()
            .unwrap()
            .destination = BlockValueId(1);
        reuse_finite_padding(&mut program);
        reuse_finite_padding(&mut mixed);
        reuse_finite_padding(&mut overwritten);
        assert_eq!(program.tiles[0].work.len(), 1);
        assert_eq!(mixed.tiles[0].work.len(), 2);
        assert_eq!(overwritten.tiles[0].work.len(), 2);
    }
}
