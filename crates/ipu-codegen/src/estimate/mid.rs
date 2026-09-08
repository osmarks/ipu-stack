//! Coarse costing of whole-device primitives. Work scales with the selected
//! tensor program and axis partitions, not the number of tile IR objects.

use super::*;
use crate::{MidOperationKind, MidProgram, Primitive, TileKernelSpec};

pub(crate) fn analyze(
    program: &MidProgram,
    copies: &BTreeMap<MidValueId, u32>,
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
            MidOperationKind::Primitive(Primitive::Compute {
                reuse_input: Some(input),
                ..
            }) => alias(operation.results[0], operation.inputs[*input]),
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
    let mut bytes = vec![0u64; parent.len()];
    let mut classes = vec![MemoryClass::Ipu21Standard; parent.len()];
    for value in &program.values {
        let id = roots[value.id.index() as usize];
        let class = value.tensor_type.format.layout.memory_class;
        let mut size = maximum_shard_bytes(&value.tensor_type).checked_add(tail[id])?;
        if element[id] {
            let alignment = u64::from(if class == MemoryClass::Ipu21Interleaved {
                ipu_package::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_package::TILE_MEMORY_ELEMENT_SIZE
            });
            size = size.div_ceil(alignment).checked_mul(alignment)?;
        }
        size = size.checked_mul(u64::from(copies.get(&value.id).copied().unwrap_or(1)))?;
        bytes[id] = bytes[id].max(size);
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
    let mut live = vec![false; parent.len()];
    for input in &program.inputs {
        live[roots[input.value.index() as usize]] = true;
    }
    let mut cycles = ProgramCycles::default();
    let mut peak = MemoryPeaks::default();
    let mut rows = 0u64;
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
        let mut usage = scratch;
        let mut maximum_standard = 0;
        for id in 0..live.len() {
            if live[id] {
                usage.add_class(classes[id], bytes[id]);
                if classes[id] == MemoryClass::Ipu21Standard {
                    maximum_standard = maximum_standard.max(bytes[id]);
                }
            }
        }
        peak.observe(usage, maximum_standard.max(scratch.standard));
        for id in 0..live.len() {
            if last[id] <= index {
                live[id] = false;
            }
        }
    }
    peak.exchange_rows = rows;
    peak.standard = peak.standard.saturating_add(rows);
    peak.total = peak.total.saturating_add(rows);
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
            price.total =
                super::primitive::kernel_cycles(&kernel, &inputs, &out).saturating_mul(calls);
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
                let (exchange, footprint) =
                    exchange_price(bytes.max(sends), 1, movement_fragment_bytes(input, output));
                price.exchange = exchange;
                rows = footprint;
            }
            price.total = price
                .exchange
                .saturating_add(bytes.div_ceil(IPU21_TARGET_COSTS.local_copy_bytes_per_cycle))
                .saturating_add(IPU21_TARGET_COSTS.local_copy_call_cycles);
            if input.format.layout.order != output.format.layout.order
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
                price.total = price.exchange.saturating_add(super::primitive::cast_cycles(
                    input.format.precision,
                    output.format.precision,
                    bytes.div_ceil(output.format.precision.bytes()),
                    output.format.layout.order.fp8_cast_panel_rows(
                        out.shape.0[..out.shape.0.len().saturating_sub(1)]
                            .iter()
                            .map(|&n| u64::from(n))
                            .product(),
                        u64::from(*out.shape.0.last().unwrap_or(&1)),
                    ),
                ));
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
    if bytes == 0 || phases == 0 {
        return (0, 0);
    }
    // Coarse useful-payload assumption for blocked tensor movement. Placement
    // and physical scheduling later determine the actual fragmentation.
    let fragments = bytes.div_ceil(fragment_bytes.max(1)).max(phases);
    let cycles = bytes
        .div_ceil(IPU21_TARGET_COSTS.exchange_bytes_per_cycle)
        .max(fragments.saturating_mul(IPU21_LOGICAL_FRAGMENT_CYCLES))
        .saturating_add(phases.saturating_mul(IPU21_TARGET_COSTS.exchange_phase_cycles));
    let rows = ExchangeFootprint {
        phases,
        maximum_transfer_chunks_per_tile: fragments,
    }
    .estimated_row_bytes();
    (cycles, rows)
}

pub(crate) fn region_peak_memory(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MemoryPeaks {
    region_peak_memory_with_multiplicity(initial, operations, outputs, values, &BTreeMap::new())
}

pub(crate) fn region_peak_memory_with_multiplicity(
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> MemoryPeaks {
    region_estimate(
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
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
    allocation_multiplicity: &BTreeMap<MidValueId, u32>,
) -> Option<(ProgramCycles, MemoryPeaks)> {
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
        tile_count: values
            .iter()
            .map(|value| value.tensor_type.format.layout.tiling.tile_count)
            .max()
            .unwrap_or(1),
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

    let program = crate::mid::implementation::resolve(candidate)?;
    analyze(&program, allocation_multiplicity)
}
