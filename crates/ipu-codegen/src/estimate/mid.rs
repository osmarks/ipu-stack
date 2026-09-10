//! Coarse costing of whole-device primitives. Work scales with the selected
//! tensor program and axis partitions, not the number of tile IR objects.

use super::*;
use crate::{MidOperationKind, MidProgram, Primitive, TileKernelSpec};

pub(crate) fn analyze(
    program: &MidProgram,
    copies: &BTreeMap<MidValueId, u32>,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    analyze_observed(program, copies, &mut ())
}

/// The normal estimator uses the zero-cost observer. Diagnostics record the
/// same allocation and liveness decisions, without a second accounting model.
pub(super) trait MemoryObserver {
    fn value(
        &mut self,
        _value: &MidValue,
        _root: usize,
        _shard_bytes: u64,
        _aligned_bytes: u64,
        _copies: u32,
        _bytes: u64,
        _tile_shards: &[u64],
        _tile_bytes: &[u64],
    ) {
    }
    fn step(
        &mut self,
        _index: usize,
        _operation: &MidOperation,
        _count: u64,
        _live: &[bool],
        _scratch: MemoryUsage,
        _usage: MemoryUsage,
        _tile_usage: &[MemoryUsage],
        _tile_scratch: &[MemoryUsage],
    ) {
    }
}
impl MemoryObserver for () {}

pub(super) fn analyze_observed(
    program: &MidProgram,
    copies: &BTreeMap<MidValueId, u32>,
    observer: &mut impl MemoryObserver,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    analyze_storage::<true>(program, copies, observer)
}

/// A fitting upper bound needs no refinement. Every failed capacity screen is
/// checked on actual owners before it can discard a candidate.
fn analyze_with_budget(
    program: &MidProgram,
    copies: &BTreeMap<MidValueId, u32>,
    config: &crate::PipelineConfig,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    let bound = analyze_storage::<false>(program, copies, &mut ())?;
    if bound.1.fits_ipu21_with_budget(
        config.standard_memory_reservation_bytes,
        config.tile_memory_budget_bytes,
    ) {
        Some(bound)
    } else {
        analyze(program, copies)
    }
}

fn analyze_storage<const PER_TILE: bool>(
    program: &MidProgram,
    copies: &BTreeMap<MidValueId, u32>,
    observer: &mut impl MemoryObserver,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    let mut steps = Vec::new();
    fn flatten<'a>(
        operations: &'a [MidOperation],
        count: u64,
        steps: &mut Vec<(&'a MidOperation, u64)>,
    ) {
        for operation in operations {
            if let MidOperationKind::Repeat(repeat) = &operation.kind {
                flatten(
                    &repeat.body.operations,
                    count.saturating_mul(u64::from(repeat.count)),
                    steps,
                );
            }
            steps.push((operation, count));
        }
    }
    flatten(&program.operations, 1, &mut steps);
    let mut parent = (0..program.values.len()).collect::<Vec<_>>();
    fn root(parent: &[usize], mut id: usize) -> usize {
        while parent[id] != id {
            id = parent[id];
        }
        id
    }
    let mut alias = |a: MidValueId, b: MidValueId| {
        let a = root(&parent, a.index() as usize);
        let b = root(&parent, b.index() as usize);
        parent[a.max(b)] = a.min(b);
    };
    for (operation, _) in &steps {
        match &operation.kind {
            MidOperationKind::Primitive(Primitive::Compute { output_aliases, .. }) => {
                for &(output, input) in output_aliases {
                    alias(operation.results[output], operation.inputs[input]);
                }
            }
            MidOperationKind::Repeat(repeat) => {
                for (&argument, &input) in repeat.body.arguments.iter().zip(&operation.inputs) {
                    alias(argument, input);
                }
                for (&result, &input) in operation.results.iter().zip(&operation.inputs) {
                    alias(result, input);
                }
                for (&argument, values) in repeat
                    .body
                    .arguments
                    .iter()
                    .skip(operation.inputs.len())
                    .zip(&repeat.iterated_inputs)
                {
                    alias(argument, *values.first()?);
                }
            }
            _ => {}
        }
    }
    let roots = (0..parent.len())
        .map(|id| root(&parent, id))
        .collect::<Vec<_>>();
    let mut element = vec![false; parent.len()];
    let mut tail = vec![0; parent.len()];
    for (operation, _) in &steps {
        if let MidOperationKind::Primitive(Primitive::Compute {
            kernel: TileKernelSpec::Gemm { multiply, .. },
            ..
        }) = &operation.kind
        {
            element[roots[operation.inputs[0].index() as usize]] = true;
            element[roots[operation.results[0].index() as usize]] = true;
            tail[roots[operation.inputs[0].index() as usize]] = 8 * multiply.bytes();
        }
    }
    let tiles = if PER_TILE {
        usize::from(program.tile_count)
    } else {
        1
    };
    let mut bytes = vec![vec![0u64; tiles]; parent.len()];
    let mut classes = vec![MemoryClass::Ipu21Standard; parent.len()];
    for value in &program.values {
        let id = roots[value.id.index() as usize];
        let class = value.tensor_type.format.layout.memory_class;
        let layout = &value.tensor_type.format.layout;
        let resolved = layout.resolve(&value.tensor_type.shape).ok()?;
        if layout.tiling.tile_count > program.tile_count || program.tile_count == 0 {
            return None;
        }
        let count = copies.get(&value.id).copied().unwrap_or(1);
        let alignment = if element[id] {
            u64::from(if class == MemoryClass::Ipu21Interleaved {
                ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_package::TILE_MEMORY_ELEMENT_SIZE
            })
        } else {
            1
        };
        let mut tile_shards = vec![0; tiles];
        let mut tile_bytes = vec![0; tiles];
        let mut aligned_bytes = 0;
        let owners = if PER_TILE {
            layout.tiling.tile_count
        } else {
            1
        };
        for owner in 0..owners {
            let elements = if PER_TILE {
                resolved.tile_elements(owner)
            } else {
                resolved.maximum_tile_elements()
            };
            let shard = elements.checked_mul(value.tensor_type.format.precision.bytes())?;
            if shard == 0 {
                continue;
            }
            let tile = (usize::from(owner) + usize::from(value.tile_offset)) % tiles;
            let size = shard
                .checked_add(tail[id])?
                .div_ceil(alignment)
                .checked_mul(alignment)?;
            aligned_bytes = aligned_bytes.max(size);
            tile_shards[tile] = shard;
            tile_bytes[tile] = size.checked_mul(u64::from(count))?;
            bytes[id][tile] = bytes[id][tile].max(tile_bytes[tile]);
        }
        observer.value(
            value,
            id,
            *tile_shards.iter().max()?,
            aligned_bytes,
            count,
            *tile_bytes.iter().max()?,
            &tile_shards,
            &tile_bytes,
        );
        classes[id] = class;
    }
    let mut last = vec![0usize; parent.len()];
    for (index, (operation, _)) in steps.iter().enumerate() {
        for value in operation.inputs.iter().chain(&operation.results) {
            last[roots[value.index() as usize]] = index;
        }
        if let MidOperationKind::Repeat(repeat) = &operation.kind {
            for value in repeat.iterated_inputs.iter().flatten() {
                last[roots[value.index() as usize]] = index;
            }
        }
    }
    for value in &program.outputs {
        last[roots[value.index() as usize]] = steps.len();
    }
    // A region argument represents a resident sequence, not just this
    // iteration's block. Its later members remain needed after the local use.
    for (&value, &count) in copies {
        if count > 1 {
            last[roots[value.index() as usize]] = steps.len();
        }
    }
    let mut live = vec![false; parent.len()];
    for input in &program.inputs {
        live[roots[input.value.index() as usize]] = true;
    }
    let mut cycles = ProgramCycles::default();
    let mut peak = MemoryPeaks::default();
    let mut rows = 0u64;
    let mut accounted = vec![false; live.len()];
    let mut tile_live = vec![MemoryUsage::default(); tiles];
    let maximum_sizes = bytes
        .iter()
        .map(|sizes| sizes.iter().copied().max().unwrap_or(0))
        .collect::<Vec<_>>();

    for (index, (operation, count)) in steps.into_iter().enumerate() {
        for value in operation.inputs.iter().chain(&operation.results) {
            live[roots[value.index() as usize]] = true;
        }
        let (price, scratch, row_bytes) = operation_cost(operation, &program.values)?;
        cycles.total = cycles
            .total
            .saturating_add(price.total.saturating_mul(count));
        cycles.exchange = cycles
            .exchange
            .saturating_add(price.exchange.saturating_mul(count));
        rows = rows.saturating_add(row_bytes);
        let maximum_standard = (0..live.len())
            .filter(|&id| live[id] && classes[id] == MemoryClass::Ipu21Standard)
            .map(|id| maximum_sizes[id])
            .max()
            .unwrap_or(0);
        // Update only roots whose liveness changed since the preceding step.
        for id in 0..live.len() {
            if live[id] == accounted[id] {
                continue;
            }
            for (usage, &size) in tile_live.iter_mut().zip(&bytes[id]) {
                if live[id] {
                    usage.add_class(classes[id], size);
                } else {
                    let used = match classes[id] {
                        MemoryClass::Ipu21Standard => &mut usage.standard,
                        MemoryClass::Ipu21Interleaved => &mut usage.interleaved,
                    };
                    *used = used.checked_sub(size)?;
                }
            }
            accounted[id] = live[id];
        }
        // Reduction and conversion scratch in operation_cost is proportional
        // to the output shard, and resides on that output's owners.
        let mut tile_scratch = vec![MemoryUsage::default(); tiles];
        if !PER_TILE {
            tile_scratch[0] = scratch;
        } else if scratch.total() != 0 {
            let output = &program.values[operation.results.first()?.index() as usize];
            let layout = &output.tensor_type.format.layout;
            let resolved = layout.resolve(&output.tensor_type.shape).ok()?;
            let maximum = resolved.maximum_tile_elements();
            for owner in 0..layout.tiling.tile_count {
                let elements = resolved.tile_elements(owner);
                let tile = (usize::from(owner) + usize::from(output.tile_offset)) % tiles;
                tile_scratch[tile] = MemoryUsage {
                    standard: scratch.standard.checked_mul(elements)?.div_ceil(maximum),
                    interleaved: scratch.interleaved.checked_mul(elements)?.div_ceil(maximum),
                };
            }
        }
        let tile_usage = tile_live
            .iter()
            .zip(&tile_scratch)
            .map(|(&live, &scratch)| live.saturating_add(scratch))
            .collect::<Vec<_>>();
        let usage = tile_usage
            .iter()
            .copied()
            .max_by_key(|usage| usage.total())
            .unwrap_or_default();
        observer.step(
            index,
            operation,
            count,
            &live,
            scratch,
            usage,
            &tile_usage,
            &tile_scratch,
        );
        for &usage in &tile_usage {
            peak.observe(usage, maximum_standard.max(scratch.standard));
        }
        for id in 0..live.len() {
            if last[id] <= index {
                live[id] = false;
            }
        }
    }
    peak.exchange_rows = rows;
    Some((cycles, peak))
}

fn local_tensor(tensor: &TensorType) -> Option<TensorType> {
    let resolved = tensor.format.layout.resolve(&tensor.shape).ok()?;
    let shape = if let Some(axes) = resolved.axes() {
        TensorShape(axes.iter().map(|axis| axis.maximum_extent()).collect())
    } else {
        TensorShape(vec![u32::try_from(resolved.maximum_tile_elements()).ok()?])
    };
    Some(TensorType {
        shape,
        format: tensor.format.clone(),
    })
}

pub(crate) fn operation_cost(
    operation: &MidOperation,
    values: &[MidValue],
) -> Option<(ProgramCycles, MemoryUsage, u64)> {
    let tensor = |id: MidValueId| &values[id.index() as usize].tensor_type;
    if matches!(operation.kind, MidOperationKind::Repeat(_)) {
        return Some((ProgramCycles::default(), MemoryUsage::default(), 0));
    }
    let output = tensor(*operation.results.first()?);
    let mut out = local_tensor(output)?;
    let mut scratch = MemoryUsage::default();
    let mut rows = 0;
    let mut price = ProgramCycles::default();
    match &operation.kind {
        MidOperationKind::Primitive(Primitive::Compute {
            kernel,
            operands,
            product,
            ..
        }) => {
            let mut inputs = operation
                .inputs
                .iter()
                .zip(operands)
                .map(|(&id, window)| {
                    let mut local = local_tensor(tensor(id))?;
                    for &(axis, start, end) in &window.0 {
                        local.shape.0[usize::from(axis)] =
                            local.shape.0[usize::from(axis)].min(end.checked_sub(start)?);
                    }
                    Some(local)
                })
                .collect::<Option<Vec<_>>>()?;
            let mut kernel = kernel.clone();
            let mut calls = 1u64;
            if let Some(axes) = product {
                let left_axis = axes.left_inner.resolve(inputs[0].shape.0.len()).ok()?;
                let right_axis = axes.right_inner.resolve(inputs[1].shape.0.len()).ok()?;
                let column_axis = axes.output_column.resolve(out.shape.0.len()).ok()?;
                if let TileKernelSpec::Gemm {
                    inner_block,
                    output_columns,
                    ..
                } = &mut kernel
                {
                    if *inner_block == 0 || *output_columns == 0 {
                        return None;
                    }
                    if let Some(group) = out.format.layout.order.gemm_output_group() {
                        *output_columns = (*output_columns).min(group);
                    }
                    calls = u64::from(inputs[0].shape.0[left_axis].div_ceil(*inner_block))
                        .checked_mul(u64::from(
                            out.shape.0[column_axis].div_ceil(*output_columns),
                        ))?;
                    *inner_block = (*inner_block).min(inputs[0].shape.0[left_axis]);
                    *output_columns = (*output_columns).min(out.shape.0[column_axis]);
                    inputs[0].shape.0[left_axis] = *inner_block;
                    inputs[1].shape.0[right_axis] = *inner_block;
                    out.shape.0[column_axis] = *output_columns;
                }
            }
            price.total = super::primitive::kernel_cycles(
                &kernel,
                |i| inputs.get(i).map(super::primitive::Geometry::Tensor),
                super::primitive::Geometry::Tensor(&out),
            )
            .saturating_mul(calls);
        }
        MidOperationKind::Primitive(Primitive::Sum { axis, staging }) => {
            let contributors = u64::from(tensor(operation.inputs[0]).shape.0[usize::from(*axis)]);
            let remote = contributors.saturating_sub(1);
            let bytes = maximum_shard_bytes(output);
            let per_stage = staging.remote_partials_per_stage(remote);
            let stages = remote.div_ceil(per_stage);
            scratch.standard = bytes.saturating_mul(per_stage.saturating_add(2));
            let (exchange, footprint) = exchange_price(bytes.saturating_mul(remote), stages, 256);
            price.exchange = exchange;
            rows = footprint;
            let elements = bytes.div_ceil(output.format.precision.bytes());
            let full_stages = remote / per_stage;
            let tail = remote % per_stage;
            price.total = exchange
                .saturating_add(full_stages.saturating_mul(
                    crate::kernel::cost::f16_reduction_cycles(elements, per_stage + 1),
                ))
                .saturating_add(if tail == 0 {
                    0
                } else {
                    crate::kernel::cost::f16_reduction_cycles(elements, tail + 1)
                });
        }
        MidOperationKind::Primitive(Primitive::Copy { .. }) | MidOperationKind::Convert(_) => {
            let input = tensor(operation.inputs[0]);
            let bytes = maximum_shard_bytes(output);
            let local_conversion = matches!(&operation.kind, MidOperationKind::Convert(plan)
                if plan.strategy == crate::ConversionStrategy::LocalKernel);
            let same_ownership = local_conversion
                || (crate::mid::implementation::same_distribution(input, output)
                    && values[operation.inputs[0].index() as usize].tile_offset
                        == values[operation.results[0].index() as usize].tile_offset);
            if !same_ownership
                || matches!(
                    operation.kind,
                    MidOperationKind::Primitive(Primitive::Copy {
                        mapping: crate::CoordinateMapping { view: Some(_), .. },
                        ..
                    })
                )
            {
                let destinations = u64::from(output.format.layout.tiling.tile_count);
                let sources = u64::from(input.format.layout.tiling.tile_count).max(1);
                let sends = bytes
                    .saturating_mul(destinations)
                    .div_ceil(sources)
                    .min(maximum_shard_bytes(input));
                let payload = bytes.max(sends);
                let identity = !matches!(&operation.kind,
                    MidOperationKind::Primitive(Primitive::Copy { mapping, .. })
                    if *mapping != crate::CoordinateMapping::default());
                let fragments = identity
                    .then(|| super::movement::grid_fragments(input, output))
                    .flatten()
                    .unwrap_or_else(|| {
                        payload.div_ceil(movement_fragment_bytes(input, output).max(1))
                    });
                let (exchange, footprint) = exchange_fragment_price(payload, 1, fragments);
                price.exchange = exchange;
                rows = footprint;
            }
            price.total = price
                .exchange
                .saturating_add(bytes.div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle))
                .saturating_add(IPU21_TARGET_COSTS.local_copy_call_cycles);
            if input.format.precision == output.format.precision
                && input.format.layout.order != output.format.layout.order
                && !input.format.supports_micro_panel_exchange(&output.format)
            {
                scratch.standard = bytes;
                let elements = bytes.div_ceil(output.format.precision.bytes());
                price.total = price.total.saturating_add(
                    if output.format.layout.order == ElementOrder::RowMajor {
                        elements.saturating_mul(10)
                    } else {
                        row_major_pack_cycles(&out, elements)
                    },
                );
            }
            if input.format.precision != output.format.precision {
                price.total = price
                    .exchange
                    .saturating_add(Ipu21CostModel.cast_format_cycles(input, &output.format));
            }
        }
        MidOperationKind::Operator { .. } => return None,
        MidOperationKind::Repeat(_) => unreachable!(),
    }
    Some((price, scratch, rows))
}

/// Packed matrices with linear ownership cross row/panel boundaries. Use their
/// native column grain as a conservative fragment size, rather than treating
/// these mappings like long contiguous blocked transfers. No tile enumeration.
fn movement_fragment_bytes(input: &TensorType, output: &TensorType) -> u64 {
    if input.format.layout.tiling != output.format.layout.tiling
        && [input, output]
            .iter()
            .any(|t| t.format.layout.tiling.linear_grain().is_some())
    {
        return [input, output]
            .into_iter()
            .filter_map(|t| {
                (t.format.layout.order != ElementOrder::RowMajor)
                    .then(|| {
                        t.format
                            .layout
                            .order
                            .retained_linear_column_grain(t.format.precision)
                    })
                    .flatten()
                    .map(|grain| u64::from(grain) * t.format.precision.bytes())
            })
            .min()
            .unwrap_or(256)
            .min(256);
    }
    256
}

fn exchange_price(bytes: u64, phases: u64, fragment_bytes: u64) -> (u64, u64) {
    exchange_fragment_price(bytes, phases, bytes.div_ceil(fragment_bytes.max(1)))
}

pub(crate) fn region_peak_memory(
    config: &crate::PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(
        config,
        initial,
        operations,
        outputs,
        values,
        &BTreeMap::new(),
    )
}

pub(crate) fn region_peak_memory_with_multiplicity(
    config: &crate::PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> MemoryPeaks {
    region_estimate(
        config,
        initial,
        operations,
        outputs,
        values,
        allocation_multiplicity,
    )
    .map_or_else(unavailable_memory, |(_, peak)| peak)
}

pub(crate) fn unavailable_memory() -> MemoryPeaks {
    MemoryPeaks {
        standard: u64::MAX,
        interleaved: u64::MAX,
        total: u64::MAX,
        ..MemoryPeaks::default()
    }
}

pub(crate) fn region_estimate(
    config: &crate::PipelineConfig,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    let program = resolved_region(config.tile_count, initial, operations, outputs, values)?;
    analyze_with_budget(&program, allocation_multiplicity, config)
}

pub(super) fn resolved_region(
    tile_count: u16,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> Option<MidProgram> {
    let mut outputs = outputs.to_vec();
    // A pending view still needs its source storage at the region boundary.
    for operation in operations.iter().rev() {
        if let Some(offer) = operation
            .operator_plan()
            .and_then(|plan| plan.deferred_output)
            && operation
                .results
                .iter()
                .any(|result| outputs.contains(result))
        {
            outputs.push(operation.inputs[offer.source_input]);
        }
    }
    let candidate = crate::MidProgram {
        tile_count,
        inputs: initial
            .iter()
            .map(|&value| crate::MidInput {
                name: String::new(),
                kind: crate::GraphInputKind::Host,
                value,
            })
            .collect(),
        values: values.to_vec(),
        operations: operations.to_vec(),
        outputs,
        ..crate::MidProgram::default()
    };

    crate::mid::implementation::resolve(candidate)
}
