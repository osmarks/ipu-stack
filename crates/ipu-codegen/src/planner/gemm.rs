//! GEMM alternatives are explicit M/N/K grids. Preparation, independent partial
//! products, and their reduction are emitted here, before costing or lowering.

use super::candidates::{BoundaryValue, Candidate};
use super::construction::{copy, value};
use super::{PlanningError, PlanningResult};
use crate::graph::{GemmOptions, HighGraph, ValueId};
use crate::kernel::{AccumulationPrecision, GemmAxes, GemmKernelMode, GemmWeightLoad};
use crate::mid::{MidOperation, MidOperationKind, OperandIndexing};
use crate::{
    AmpOrder, AxisTiling, BlockMajorOrder, ElementOrder, Layout, MemoryClass, OwnerMap, Padding,
    PipelineConfig, Precision, TensorAxis, TensorFormat, TensorShape, TensorTiling, TensorType,
};
use rayon::prelude::*;

// Both supported multiply precisions currently produce FP16 mid values.
pub(super) const OUTPUT_PRECISION: Precision = Precision::F16;

fn partitions(extent: u32, tiles: u16) -> Vec<u16> {
    let maximum = extent.min(u32::from(tiles)) as u16;
    let mut result = Vec::new();
    let mut previous = 0;
    for n in 1..=maximum {
        let width = extent.div_ceil(u32::from(n));
        if width != previous {
            result.push(n);
            previous = width;
        }
    }
    result
}

// Enumerate geometry only. Cost tradeoffs belong to search, where boundary
// conversions and live tensors are available.
fn grids([m, n, k]: [u32; 3], tiles: u16) -> Vec<[u16; 3]> {
    let mut grids = Vec::new();
    for rows in partitions(m, tiles) {
        for columns in partitions(n, tiles / rows) {
            for inner in partitions(k, tiles / rows / columns) {
                grids.push([rows, columns, inner]);
            }
        }
    }
    grids
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

#[tracing::instrument(name = "gemm_candidates", skip_all, fields(position))]
pub(super) fn generate(
    high: &HighGraph,
    position: usize,
    tensors: &std::collections::BTreeMap<ValueId, TensorType>,
    config: &PipelineConfig,
    options: GemmOptions,
    offers: &std::collections::BTreeMap<ValueId, std::collections::BTreeSet<Layout>>,
) -> PlanningResult<Vec<Candidate>> {
    let op = &high.operations()[position];
    let origin = op.results[0];
    let shape = high.value_shape(origin).unwrap();
    let sources = [&tensors[&op.inputs[0]], &tensors[&op.inputs[1]]];
    let assignments = choices(sources, shape, options, config.tile_count)?
        .into_iter()
        .filter(|choice| {
            // Each GEMM operand/result is a real tile-local allocation. A single
            // one exceeding total tile capacity is infeasible regardless of
            // boundaries, other live values, or placement. This is not a rank.
            choice
                .operands
                .iter()
                .chain([&choice.result])
                .all(|tensor| {
                    tensor
                        .format
                        .layout
                        .resolve(&tensor.shape)
                        .is_ok_and(|layout| {
                            layout.maximum_tile_elements() * tensor.format.precision.bytes()
                                <= config
                                    .tile_memory_budget_bytes
                                    .min(config.target.planned_data_bytes())
                        })
                })
        })
        .filter(|choice| {
            // These boundaries require redistribution. Copies cannot yet gather
            // an odd FP16 half-word tail from a distributed packed result.
            choice.result.format.layout.tiling.tile_count == 1
                || choice
                    .result
                    .format
                    .layout
                    .shard_extents(&choice.result.shape)
                    .is_ok_and(|shards| {
                        shards.iter().all(|(_, extents)| {
                            extents
                                .iter()
                                .map(|e| u64::from(e.logical_end - e.start))
                                .product::<u64>()
                                .is_multiple_of(2)
                        })
                    })
        })
        .collect::<Vec<_>>();
    let boundaries = offers[&op.inputs[0]]
        .iter()
        .flat_map(|left| {
            offers[&op.inputs[1]].iter().flat_map(move |right| {
                offers[&origin]
                    .iter()
                    .map(move |output| (left, right, output))
            })
        })
        .filter(|(left, right, _)| op.inputs[0] != op.inputs[1] || left == right)
        .collect::<Vec<_>>();
    // Stream implementation graphs into the common frontier rather than retaining
    // hundreds of thousands of graphs before pruning the fixed boundary groups.
    let frontiers = boundaries
        .into_par_iter()
        .map(|(left, right, output)| {
            let mut operands = super::candidates::LiveValues::new();
            for (i, &id) in op.inputs.iter().enumerate() {
                let mut tensor = sources[i].clone();
                tensor.format.layout = [left, right][i].clone();
                // A repeated high operand imports one representation; append
                // prepares the second role internally when its layout differs.
                operands.entry(id).or_insert(BoundaryValue {
                    tensor,
                    owners: OwnerMap::default(),
                });
            }
            let template = Candidate::inputs(high, &operands, config.tile_count, position + 1);
            let inputs = [
                template.bindings[&op.inputs[0]],
                template.bindings[&op.inputs[1]],
            ];
            // Dominance within a fixed boundary is compositional. Prune chunks
            // independently, then merge their frontiers with the same search code.
            // Chunking bounds worker memory; it does not limit the geometry search.
            let chunks = assignments
                .par_chunks(4096)
                .map(|assignments| {
                    let candidates = assignments.iter().map(|choice| {
                        let mut candidate = template.clone();
                        let result = append(
                            &mut candidate.graph,
                            inputs,
                            choice,
                            options,
                            shape,
                            op.id,
                            origin,
                            Some(output),
                        );
                        candidate.bindings.insert(origin, result);
                        candidate.graph.outputs = vec![result];
                        candidate
                    });
                    super::search::prune(candidates, config)
                })
                .collect::<PlanningResult<Vec<_>>>()?;
            super::search::prune(chunks.into_iter().flatten(), config)
        })
        .collect::<PlanningResult<Vec<_>>>()?;
    Ok(frontiers.into_iter().flatten().collect())
}

/// One enumerated distributed GEMM assignment; no high bindings or search state.
pub(super) struct GemmChoice {
    /// Kernel operand order; swapped multiplication reverses the high inputs.
    pub operands: [TensorType; 2],
    result: TensorType,
    reduction: Option<Layout>,
    swapped: bool,
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
    let k =
        sources[0].shape.0[sources[0].shape.0.len() - if options.transpose_left { 2 } else { 1 }];
    let grain = if precision == Precision::F16 { 16 } else { 32 };
    let mut candidates = Vec::new();
    let mut batches = vec![(1u16, Vec::new())];
    for axis in 0..rank - 2 {
        // Copies cannot exchange fractions of a word at a batch boundary.
        // Keep whole groups of matrices where a matrix has a sub-word tail.
        let mut batch_grain = 1;
        for (s, bytes) in sources
            .iter()
            .map(|s| (&s.shape, s.format.precision.bytes()))
        {
            if let Some(a) = (axis + s.0.len()).checked_sub(rank)
                && s.0[a] != 1
            {
                let plane = s.0[a + 1..].iter().map(|&n| u64::from(n)).product::<u64>() * bytes;
                batch_grain = batch_grain.max(if plane.is_multiple_of(4) {
                    1
                } else if plane.is_multiple_of(2) {
                    2
                } else {
                    4
                });
            }
        }
        let blocks = if shape.0[axis].is_multiple_of(batch_grain) {
            shape.0[axis] / batch_grain
        } else {
            1
        };
        batches = batches
            .into_iter()
            .flat_map(|(used, axes)| {
                partitions(blocks, tile_count / used)
                    .into_iter()
                    .map(move |parts| {
                        let mut axes = axes.clone();
                        axes.push(AxisTiling::new(
                            TensorAxis::FromEnd((rank - axis) as u16),
                            parts,
                            if parts == 1 { 1 } else { batch_grain },
                            Padding::Reject,
                        ));
                        (used * parts, axes)
                    })
            })
            .collect();
    }
    for swapped in [false, true] {
        let (sources, transpose) = if swapped {
            (
                [sources[1], sources[0]],
                [!options.transpose_right, !options.transpose_left],
            )
        } else {
            (sources, [options.transpose_left, options.transpose_right])
        };
        let m = shape.0[rank - if swapped { 1 } else { 2 }];
        let n = shape.0[rank - if swapped { 2 } else { 1 }];
        // Swapping puts logical output columns on the kernel's row axis.
        // Word-aligned boundaries permit subsequent row-major redistribution.
        let row_grain = if swapped { 2 } else { 1 };
        for (groups, batch_axes) in &batches {
            for [rows, columns, inner] in grids(
                [m.div_ceil(row_grain), n.div_ceil(16), k.div_ceil(grain)],
                tile_count / groups,
            ) {
                let tiles = rows * columns * inner * groups;
                let kw = k.div_ceil(grain).div_ceil(u32::from(inner)) * grain;
                let nw = n.div_ceil(16).div_ceil(u32::from(columns)) * 16;
                if kw > u32::from(u16::MAX) {
                    continue;
                }
                let mut orders = Vec::new();
                for order in [
                    [1, 2, 0],
                    [1, 0, 2],
                    [0, 2, 1],
                    [0, 1, 2],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut strides = [0; 3];
                    let mut stride = 1;
                    for axis in order {
                        strides[axis] = if [rows, columns, inner][axis] == 1 {
                            1
                        } else {
                            stride
                        };
                        stride *= [rows, columns, inner][axis];
                    }
                    if !orders.contains(&strides) {
                        orders.push(strides);
                    }
                }
                for strides in orders {
                    let mut operands = std::array::from_fn(|i| {
                        TensorType::new(
                            sources[i].shape.0.clone(),
                            precision,
                            operand_layout(
                                i == 0,
                                transpose[i],
                                rows * columns * inner,
                                [rows, columns][i],
                                inner,
                                [columns, rows][i],
                                kw,
                                nw,
                            ),
                        )
                    });
                    let mut result = TensorType::new(
                        shape.0.clone(),
                        OUTPUT_PRECISION,
                        operand_layout(true, swapped, tiles, rows, columns, 1, nw, nw),
                    );
                    if kw > grain {
                        result.format.layout.memory_class = MemoryClass::Ipu21Interleaved;
                    }
                    for (operand, tensor) in operands
                        .iter_mut()
                        .chain(std::iter::once(&mut result))
                        .enumerate()
                    {
                        let tiling = &mut tensor.format.layout.tiling;
                        if operand != 1 {
                            tiling.axes[0].block_size = row_grain;
                            tiling.axes[0].padding_multiple = row_grain;
                        }
                        tiling.axes[0].tile_stride = Some(strides[usize::from(operand == 1)]);
                        tiling.axes[1].tile_stride =
                            Some(strides[if operand == 2 { 1 } else { 2 }]);
                        let mut stride = rows * columns * inner;
                        for axis in batch_axes {
                            let axis = axis.clone().with_tile_stride(stride);
                            stride *= axis.partitions;
                            if axis
                                .axis
                                .resolve(tensor.shape.0.len())
                                .ok()
                                .is_some_and(|a| tensor.shape.0[a] != 1)
                            {
                                tiling.axes.push(axis);
                            } else {
                                tiling.replicas *= axis.partitions;
                            }
                        }
                        tiling.tile_count = tiles;
                    }
                    let mut reductions = vec![None];
                    if inner > 1 {
                        reductions.clear();
                        // Keep a compact gather, and scatter the final reduction
                        // over either output axis when there is enough work.
                        for (rp, cp) in [
                            (rows, columns),
                            (rows * inner, columns),
                            (rows, columns * inner),
                        ] {
                            if u32::from(rp) > m || u32::from(cp) > n {
                                continue;
                            }
                            let mut layout = result.format.layout.clone();
                            layout.order = ElementOrder::RowMajor;
                            layout.memory_class = MemoryClass::Ipu21Standard;
                            layout.tiling.tile_count = rp * cp * groups;
                            layout.tiling.axes[0].partitions = rp;
                            layout.tiling.axes[1].partitions = cp;
                            // Reduction storage is row-major; it does not inherit
                            // the GEMM's padded column-panel width.
                            layout.tiling.axes[1].block_size = 1;
                            layout.tiling.axes[1].padding_multiple = 1;
                            for axis in &mut layout.tiling.axes[..2] {
                                if axis.axis == TensorAxis::FromEnd(1) {
                                    axis.block_size = 8;
                                    axis.padding_multiple = 8;
                                    axis.shard_padding_multiple = 8;
                                }
                            }
                            let row_fast = strides[0] < strides[1];
                            layout.tiling.axes[0].tile_stride = Some(if row_fast { 1 } else { cp });
                            layout.tiling.axes[1].tile_stride = Some(if row_fast { rp } else { 1 });
                            let mut stride = rp * cp;
                            for axis in &mut layout.tiling.axes[2..] {
                                axis.tile_stride = Some(stride);
                                stride *= axis.partitions;
                            }
                            if layout.resolve(shape).is_ok_and(|r| !r.has_empty_shards()) {
                                reductions.push(Some(layout));
                            }
                        }
                        result.shape.0.insert(0, u32::from(inner));
                        result.format.layout.tiling.axes.push(
                            AxisTiling::new(TensorAxis::FromStart(0), inner, 1, Padding::Reject)
                                .with_tile_stride(strides[2]),
                        );
                    }
                    for class in [MemoryClass::Ipu21Interleaved, MemoryClass::Ipu21Standard] {
                        operands[1].format.layout.memory_class = class;
                        for reduction in &reductions {
                            candidates.push(GemmChoice {
                                operands: operands.clone(),
                                result: result.clone(),
                                reduction: reduction.clone(),
                                swapped,
                                inner_block: kw,
                                output_columns: nw,
                            });
                        }
                    }
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
        result: result_type,
        reduction,
        swapped,
        inner_block: kw,
        output_columns: nw,
    } = choice;
    let (kw, nw) = (*kw, *nw);
    let inputs = if *swapped {
        [inputs[1], inputs[0]]
    } else {
        inputs
    };
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
    let mut result_type = result_type.clone();
    let transpose = if *swapped {
        [!options.transpose_right, !options.transpose_left]
    } else {
        [options.transpose_left, options.transpose_right]
    };
    let axes = GemmAxes {
        left_inner: TensorAxis::FromEnd(if transpose[0] { 2 } else { 1 }),
        right_inner: TensorAxis::FromEnd(if transpose[1] { 1 } else { 2 }),
        output_column: TensorAxis::FromEnd(if *swapped { 2 } else { 1 }),
        valid_inner: None,
        valid_columns: None,
    };
    let mut result = value(graph, origin, result_type.clone(), OwnerMap::default());
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
    if let Some(layout) = reduction {
        // Unpack on the producer tiles before distributing reduction slices.
        // This preserves padded words across exchange instead of attempting
        // to send clipped half-word tails of packed panels.
        let mut prepared = result_type.clone();
        prepared.format.layout.order = ElementOrder::RowMajor;
        prepared.format.layout.memory_class = MemoryClass::Ipu21Standard;
        for axis in &mut prepared.format.layout.tiling.axes {
            if axis.axis == TensorAxis::FromEnd(1) {
                axis.shard_padding_multiple = 8;
            }
        }
        result = copy(graph, source, result, prepared, Vec::new());
        let inner = result_type.shape.0.remove(0) as u16;
        // Row-major contributor stacks make the reduction's contiguous access
        // contract explicit. Packing/reduction fusion is a separate optimization.
        result_type.format.layout = layout.clone();
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
