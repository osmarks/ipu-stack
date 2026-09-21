//! GEMM alternatives are explicit M/N/K grids. Preparation, independent partial
//! products, and their reduction are emitted here, before costing or lowering.

use super::candidates::{BoundaryValue, Candidate};
use super::construction::{copy, value};
use super::{BoundaryLayouts, PlanningError, PlanningResult};
use crate::graph::{GemmOptions, HighGraph, ValueId};
use crate::kernel::{AccumulationPrecision, GemmAxes, GemmKernelMode, GemmWeightLoad};
use crate::mid::{MidOperation, MidOperationKind, OperandIndexing};
use crate::{
    AmpOrder, AxisTiling, BlockMajorOrder, ElementOrder, Layout, MemoryClass, OwnerMap, Padding,
    PipelineConfig, Precision, TensorAxis, TensorFormat, TensorShape, TensorTiling, TensorType,
};

fn partitions(extent: u32, tiles: u16) -> Vec<u16> {
    let maximum = extent.min(u32::from(tiles)) as u16;
    let mut result = vec![maximum];
    let mut n = 1u16;
    while n < maximum {
        result.push(n);
        let Some(next) = n.checked_mul(2) else {
            break;
        };
        n = next;
    }
    result.sort_unstable();
    result
}

/// Packed, unreplicated initial storage. Choose a balanced two-axis partition;
/// the consumer's compute grid may subsequently request replicas or regrouping.
pub(super) fn parameter_format(
    shape: &TensorShape,
    left: bool,
    transposed: bool,
    config: &PipelineConfig,
) -> TensorFormat {
    let rank = shape.0.len();
    let inner = rank - if left != transposed { 1 } else { 2 };
    let outer = if inner == rank - 1 {
        rank - 2
    } else {
        rank - 1
    };
    let mut best = None;
    for p in partitions(
        shape.0[outer].div_ceil(if left { 1 } else { 16 }),
        config.tile_count,
    ) {
        let q = (shape.0[inner].div_ceil(16)).min(u32::from(config.tile_count / p)) as u16;
        let mut layout = operand_layout(left, transposed, p * q, p, q, 1, 16, 16);
        layout.memory_class = MemoryClass::Ipu21Standard;
        let score = layout.resolve(shape).unwrap().maximum_tile_elements();
        if best.as_ref().is_none_or(|(old, _)| score < *old) {
            best = Some((score, layout));
        }
    }
    TensorFormat {
        precision: Precision::F16,
        layout: best.unwrap().1,
    }
}

fn operand_layout(
    left: bool,
    transposed: bool,
    tiles: u16,
    outer: u16,
    inner: u16,
    replicas: u16,
    k: u32,
    n: u32,
) -> Layout {
    let (outer_axis, inner_axis) = if left != transposed { (2, 1) } else { (1, 2) };
    let order = if left {
        ElementOrder::Amp(if transposed {
            AmpOrder::TransposedLeft
        } else {
            AmpOrder::Left
        })
    } else if transposed {
        ElementOrder::BlockMajor(BlockMajorOrder::TransposedMatrix {
            row_block: k as u16,
            column_block: 16,
        })
    } else {
        ElementOrder::BlockMajor(BlockMajorOrder::Matrix {
            row_block: k as u16,
            column_block: 16,
        })
    };
    Layout {
        order,
        tiling: TensorTiling {
            tile_count: tiles,
            replicas,
            axes: vec![
                AxisTiling::new(
                    TensorAxis::FromEnd(outer_axis),
                    outer,
                    if left { 1 } else { n },
                    Padding::Zero,
                )
                .with_tile_stride(if left { inner * replicas } else { 1 }),
                AxisTiling::new(TensorAxis::FromEnd(inner_axis), inner, k, Padding::Zero)
                    .with_tile_stride(if left { replicas } else { outer }),
            ],
        },
        memory_class: if left {
            MemoryClass::Ipu21Standard
        } else {
            MemoryClass::Ipu21Interleaved
        },
    }
}

pub(super) fn generate(
    high: &HighGraph,
    position: usize,
    tensors: &std::collections::BTreeMap<ValueId, TensorType>,
    layouts: &BoundaryLayouts,
    config: &PipelineConfig,
    options: GemmOptions,
    existing: &[Candidate],
) -> PlanningResult<Vec<Candidate>> {
    // The grid enumeration is intrinsic to this GEMM. Neighbour layouts do
    // not yet introduce additional direct-output implementations.
    if !existing.is_empty() {
        return Ok(Vec::new());
    }
    let op = &high.operations()[position];
    let origin = op.results[0];
    let shape = high.value_shape(origin).unwrap();
    let sources = [&tensors[&op.inputs[0]], &tensors[&op.inputs[1]]];
    let mut candidates = Vec::new();
    for choice in choices(sources, shape, options, config.tile_count)? {
        let mut operands = super::candidates::LiveValues::new();
        for (i, &id) in op.inputs.iter().enumerate() {
            let mut tensor = choice.operands[i].clone();
            tensor.format.precision = sources[i].format.precision;
            // A repeated high operand imports one representation; append
            // prepares the second role internally when its layout differs.
            operands.entry(id).or_insert(BoundaryValue {
                tensor,
                owners: OwnerMap::default(),
            });
        }
        let mut candidate = Candidate::inputs(high, &operands, config.tile_count, position + 1);
        let inputs = [
            candidate.bindings[&op.inputs[0]],
            candidate.bindings[&op.inputs[1]],
        ];
        let result = append(
            &mut candidate.graph,
            inputs,
            &choice,
            options,
            shape,
            op.id,
            origin,
            layouts.get(&origin).and_then(Option::as_ref),
        );
        candidate.bindings.insert(origin, result);
        candidate.graph.outputs = vec![result];
        if candidate.graph.validate().is_ok() {
            candidates.push(candidate);
        }
    }
    Ok(candidates)
}

/// One enumerated distributed GEMM assignment; no high bindings or search state.
pub(super) struct GemmChoice {
    pub operands: [TensorType; 2],
    grid: [u16; 3],
    inner_block: u32,
    output_columns: u32,
}

/// Enumerate assignments only. Callers decide which ones to construct and cost.
pub(super) fn choices(
    sources: [&TensorType; 2],
    shape: &TensorShape,
    options: GemmOptions,
    tile_count: u16,
) -> PlanningResult<Vec<GemmChoice>> {
    let precision = match (sources[0].format.precision, sources[1].format.precision) {
        (Precision::F16, p) | (p, Precision::F16) if p != Precision::F32 => p,
        (a @ Precision::F8F143 { .. }, b) if a == b => a,
        _ => {
            return Err(PlanningError::Unimplemented(
                "FP32 GEMM or independent FP8 operand scales",
            ));
        }
    };
    let rank = shape.0.len();
    let m = shape.0[rank - 2];
    let n = shape.0[rank - 1];
    let k =
        sources[0].shape.0[sources[0].shape.0.len() - if options.transpose_left { 2 } else { 1 }];
    let grain = if precision == Precision::F16 { 16 } else { 32 };
    let mut candidates = Vec::new();
    for rows in partitions(m, tile_count) {
        for columns in partitions(n.div_ceil(16), tile_count / rows) {
            for inner in partitions(k.div_ceil(grain), tile_count / rows / columns) {
                let tiles = rows * columns * inner;
                let kw = k.div_ceil(grain).div_ceil(u32::from(inner)) * grain;
                let nw = n.div_ceil(16).div_ceil(u32::from(columns)) * 16;
                if kw > u32::from(u16::MAX) {
                    continue;
                }
                let mut operands = [
                    TensorType {
                        shape: sources[0].shape.clone(),
                        format: TensorFormat {
                            precision,
                            layout: operand_layout(
                                true,
                                options.transpose_left,
                                tiles,
                                rows,
                                inner,
                                columns,
                                kw,
                                nw,
                            ),
                        },
                    },
                    TensorType {
                        shape: sources[1].shape.clone(),
                        format: TensorFormat {
                            precision,
                            layout: operand_layout(
                                false,
                                options.transpose_right,
                                tiles,
                                columns,
                                inner,
                                rows,
                                kw,
                                nw,
                            ),
                        },
                    },
                ];
                for class in [MemoryClass::Ipu21Interleaved, MemoryClass::Ipu21Standard] {
                    operands[1].format.layout.memory_class = class;
                    candidates.push(GemmChoice {
                        operands: operands.clone(),
                        grid: [rows, columns, inner],
                        inner_block: kw,
                        output_columns: nw,
                    });
                }
            }
        }
    }
    Ok(candidates)
}

/// Append one assignment to a caller's mid graph. Inputs may be temporaries
/// from earlier work; only provenance refers to a high operation/value.
pub(super) fn append(
    graph: &mut crate::MidGraph,
    inputs: [crate::MidValueId; 2],
    choice: &GemmChoice,
    options: GemmOptions,
    shape: &TensorShape,
    source: crate::OperationId,
    origin: ValueId,
    output_layout: Option<&Layout>,
) -> crate::MidValueId {
    let GemmChoice {
        operands,
        grid: [rows, columns, inner],
        inner_block: kw,
        output_columns: nw,
    } = choice;
    let (rows, columns, inner, kw, nw) = (*rows, *columns, *inner, *kw, *nw);
    let inputs = inputs
        .iter()
        .zip(operands)
        .map(|(id, tensor)| {
            let mut input = *id;
            let from = graph.values[input.index() as usize]
                .tensor_type
                .format
                .precision;
            let mut preparation = tensor.clone();
            preparation.format.precision = from;
            input = copy(graph, source, input, preparation, Vec::new());
            if from != tensor.format.precision {
                let output = value(
                    graph,
                    graph.values[id.index() as usize].origin,
                    tensor.clone(),
                    OwnerMap::default(),
                );
                graph.operations.push(MidOperation {
                    source: Some(source),
                    inputs: vec![input],
                    results: vec![output],
                    kind: MidOperationKind::Cast {
                        from,
                        to: tensor.format.precision,
                    },
                    operands: vec![OperandIndexing::local()],
                    output_aliases: Vec::new(),
                    output_windows: Vec::new(),
                });
                input = output;
            }
            input
        })
        .collect::<Vec<_>>();
    let rank = shape.0.len();
    let mut result_type = TensorType {
        shape: shape.clone(),
        format: TensorFormat {
            precision: Precision::F16,
            layout: Layout::amp_left_result_grid(
                nw,
                rows * columns,
                rows,
                columns,
                crate::GridOrder::ColumnsFast,
            ),
        },
    };
    let mut partial_type = result_type.clone();
    if inner > 1 {
        partial_type.shape.0.insert(0, u32::from(inner));
        partial_type.format.layout.tiling.tile_count *= inner;
        for axis in &mut partial_type.format.layout.tiling.axes {
            if axis.axis == TensorAxis::FromEnd(2) {
                axis.tile_stride = Some(columns * inner);
            }
        }
        partial_type.format.layout.tiling.axes.push(
            AxisTiling::new(TensorAxis::FromStart(0), inner, 1, Padding::Reject)
                .with_tile_stride(columns),
        );
    }
    let axes = GemmAxes {
        left_inner: TensorAxis::FromEnd(if options.transpose_left { 2 } else { 1 }),
        right_inner: TensorAxis::FromEnd(if options.transpose_right { 1 } else { 2 }),
        output_column: TensorAxis::FromEnd(1),
        valid_inner: None,
        valid_columns: None,
    };
    let mut result = value(graph, origin, partial_type, OwnerMap::default());
    graph.operations.push(MidOperation {
        source: Some(source),
        inputs,
        results: vec![result],
        kind: MidOperationKind::Gemm {
            axes,
            multiply: operands[0].format.precision,
            accumulate: if operands[0].format.precision == Precision::F16 {
                AccumulationPrecision::F32
            } else {
                AccumulationPrecision::F16
            },
            mode: GemmKernelMode::Initialize,
            weights: if operands[1].format.layout.memory_class == MemoryClass::Ipu21Interleaved {
                GemmWeightLoad::Interleaved
            } else {
                GemmWeightLoad::Standard
            },
            inner_block: kw,
            output_columns: nw,
        },
        operands: vec![OperandIndexing::local(); 2],
        output_windows: Vec::new(),
        output_aliases: Vec::new(),
    });
    if inner > 1 {
        // Row-major contributor stacks make the reduction's contiguous access
        // contract explicit. Packing/reduction fusion is a separate optimization.
        result_type.format.layout.order = ElementOrder::RowMajor;
        result_type.format.layout.memory_class = MemoryClass::Ipu21Standard;
        let mut receive = result_type.clone();
        receive.shape.0.insert(0, 1);
        let seed = copy(graph, source, result, receive.clone(), Vec::new());
        receive.shape.0[0] = u32::from(inner - 1);
        let mut offsets = vec![0; rank + 1];
        offsets[0] = 1;
        let rest = copy(graph, source, result, receive, offsets);
        let reduced = value(graph, origin, result_type.clone(), OwnerMap::default());
        graph.operations.push(MidOperation {
            source: Some(source),
            inputs: vec![seed, rest],
            results: vec![reduced],
            kind: MidOperationKind::ReductionSum { partials: inner },
            operands: vec![OperandIndexing::local(); 2],
            output_aliases: Vec::new(),
            output_windows: Vec::new(),
        });
        result = reduced;
    }
    if let Some(layout) = output_layout {
        result_type.format.layout = layout.clone();
        result = copy(graph, source, result, result_type, Vec::new());
    }
    result
}

#[cfg(test)]
#[path = "gemm_tests.rs"]
mod tests;
