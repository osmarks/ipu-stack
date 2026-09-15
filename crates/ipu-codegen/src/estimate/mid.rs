//! Coarse costing of whole-device primitives. Work scales with the selected
//! tensor program and axis partitions, not the number of tile IR objects.

use super::*;
use crate::Compute;
use crate::{MidOperationKind, MidProgram, TileKernelSpec};

/// Price complete replacement sequences with the same overflow and missing-cost rules.
pub(crate) fn operation_cycles<'a>(
    operations: impl IntoIterator<Item = &'a MidOperation>,
    values: &[MidValue],
) -> Option<u64> {
    operations.into_iter().try_fold(0u64, |sum, op| {
        operation_cost(op, values).map(|(cost, _, _)| sum.saturating_add(cost.total))
    })
}

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
pub(crate) fn analyze_with_budget(
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
            MidOperationKind::Compute(compute) => {
                let output_aliases = compute.output_aliases();
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
    for id in 0..parent.len() {
        parent[id] = root(&parent, id);
    }
    let roots = parent;
    let mut element = vec![false; roots.len()];
    let mut tail = vec![0; roots.len()];
    for (operation, _) in &steps {
        if let MidOperationKind::Compute(Compute::Product(product)) = &operation.kind {
            element[roots[operation.inputs[0].index() as usize]] = true;
            element[roots[operation.results[0].index() as usize]] = true;
            tail[roots[operation.inputs[0].index() as usize]] = 8 * product.multiply.bytes();
        }
    }
    let tiles = if PER_TILE {
        usize::from(program.tile_count)
    } else {
        1
    };
    let shifted_inputs = steps
        .iter()
        .filter_map(|(operation, _)| {
            matches!(&operation.kind, MidOperationKind::Compute(Compute::Kernel {
            kernel: TileKernelSpec::Cast { from: Precision::F16, to: Precision::F8F143 { .. } },
            output_aliases, ..
        }) if output_aliases == &[(0, 0)])
            .then(|| operation.inputs[0])
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut bytes = vec![vec![0u64; tiles]; roots.len()];
    let mut classes = vec![MemoryClass::Ipu21Standard; roots.len()];
    for value in &program.values {
        let id = roots[value.id.index() as usize];
        let class = value.tensor_type.format.layout.memory_class;
        let layout = &value.tensor_type.format.layout;
        let resolved = layout.resolve(&value.tensor_type.shape).map_err(|error| { tracing::debug!(value = ?value.id, tensor = ?value.tensor_type, %error, "invalid mid storage geometry"); }).ok()?;
        if layout.tiling.tile_count > program.tile_count || program.tile_count == 0 {
            return None;
        }
        let count = copies.get(&value.id).copied().unwrap_or(1);
        let alignment = if element[id] {
            u64::from(if class == MemoryClass::Ipu21Interleaved {
                ipu_target::ipu21::memory::IPU21_INTERLEAVED_ELEMENT_SIZE
            } else {
                ipu_target::ipu21::memory::TILE_MEMORY_ELEMENT_SIZE
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
        let mut prefixes = vec![0u64; usize::from(layout.tiling.tile_count)];
        if shifted_inputs.contains(&value.id) {
            for (owner, _) in resolved.shard_extents().ok()? {
                prefixes[usize::from(owner)] += u64::from(crate::kernel::cast::CAST_PREFIX_BYTES);
            }
        }
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
            let tile = if PER_TILE {
                usize::from(value.owners.tile(owner, program.tile_count)?)
            } else {
                0
            };
            let prefix = if PER_TILE {
                prefixes[usize::from(owner)]
            } else {
                prefixes.iter().copied().max().unwrap_or(0)
            };
            let size = shard
                .checked_add(prefix)?
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
    let mut last = vec![0usize; roots.len()];
    for (index, (operation, _)) in steps.iter().enumerate() {
        for value in operation.inputs.iter().chain(&operation.results) {
            last[roots[value.index() as usize]] = index;
        }
        if let MidOperationKind::Repeat(repeat) = &operation.kind {
            // Yields are read at the backedge, after all body work, even when
            // their producing operation had no later consumer inside the body.
            for value in repeat
                .iterated_inputs
                .iter()
                .flatten()
                .chain(&repeat.body.yields)
            {
                last[roots[value.index() as usize]] = index;
            }
        }
    }
    for value in program.outputs.iter().copied().chain(
        program
            .inputs
            .iter()
            .filter(|input| input.kind == crate::GraphInputKind::Parameter)
            .map(|input| input.value),
    ) {
        last[roots[value.index() as usize]] = steps.len();
    }
    // A region argument represents a resident sequence, not just this
    // iteration's block. Its later members remain needed after the local use.
    for (&value, &count) in copies {
        if count > 1 {
            last[roots[value.index() as usize]] = steps.len();
        }
    }
    let mut live = vec![false; roots.len()];
    for input in &program.inputs {
        live[roots[input.value.index() as usize]] = true;
    }
    let mut cycles = ProgramCycles::default();
    let mut peak = MemoryPeaks::default();
    let mut rows = 0u64;
    let mut accounted = live.clone();
    let mut tile_live = vec![MemoryUsage::default(); tiles];
    let maximum_sizes = bytes
        .iter()
        .map(|sizes| sizes.iter().copied().max().unwrap_or(0))
        .collect::<Vec<_>>();
    let mut initial_standard = 0;
    for id in (0..live.len()).filter(|&id| live[id]) {
        for (usage, &size) in tile_live.iter_mut().zip(&bytes[id]) {
            usage.add_class(classes[id], size);
        }
        if classes[id] == MemoryClass::Ipu21Standard {
            initial_standard = initial_standard.max(maximum_sizes[id]);
        }
    }
    for &usage in &tile_live {
        peak.observe(usage, initial_standard);
    }

    for (index, (operation, count)) in steps.into_iter().enumerate() {
        for value in operation.inputs.iter().chain(&operation.results) {
            live[roots[value.index() as usize]] = true;
        }
        let (price, scratch, row_bytes) = operation_cost(operation, &program.values)?;
        tracing::debug!(index, source = ?operation.source, count,
            cycles = price.total, exchange = price.exchange,
            "estimated mid operation");
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
        // Ordinary kernel scratch scales with output ownership.
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
                let tile = usize::from(output.owners.tile(owner, program.tile_count)?);
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

fn operand_tensors(
    operation: &MidOperation,
    values: &[MidValue],
    compute: &Compute,
) -> Option<Vec<TensorType>> {
    operation
        .inputs
        .iter()
        .take(compute.input_count())
        .enumerate()
        .map(|(index, &id)| {
            let mut local = local_tensor(&values[id.index() as usize].tensor_type)?;
            for &(axis, start, end) in compute
                .operand_window(index)
                .into_iter()
                .flat_map(|window| &window.0)
            {
                local.shape.0[usize::from(axis)] =
                    local.shape.0[usize::from(axis)].min(end.checked_sub(start)?);
            }
            Some(local)
        })
        .collect()
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
        MidOperationKind::Compute(
            compute @ Compute::Kernel {
                kernel,
                output_aliases,
                ..
            },
        ) => {
            let mut inputs = operand_tensors(operation, values, compute)?;
            price.total = super::primitive::kernel_cycles(
                kernel,
                |i| inputs.get(i).map(super::primitive::Geometry::Tensor),
                super::primitive::Geometry::Tensor(&out),
            );
            if matches!(
                kernel,
                TileKernelSpec::Cast {
                    from: Precision::F16,
                    to: Precision::F8F143 { .. }
                }
            ) && output_aliases == &[(0, 0)]
            {
                let chunks =
                    crate::kernel::cast::CastChunks::new(out.format.layout.order, &out.shape.0)?;
                price.total = 0;
                for (start, end) in chunks.ranges {
                    out.shape.0[chunks.axis] = end - start;
                    inputs[0].shape.0[chunks.axis] = end - start;
                    price.total += super::primitive::kernel_cycles(
                        kernel,
                        |i| inputs.get(i).map(super::primitive::Geometry::Tensor),
                        super::primitive::Geometry::Tensor(&out),
                    );
                }
            }
        }
        MidOperationKind::Compute(compute @ Compute::Product(product)) => {
            let mut inputs = operand_tensors(operation, values, compute)?;
            let axes = product.axes;
            let left_axis = axes.left_inner.resolve(inputs[0].shape.0.len()).ok()?;
            let right_axis = axes.right_inner.resolve(inputs[1].shape.0.len()).ok()?;
            let column_axis = axes.output_column.resolve(out.shape.0.len()).ok()?;
            let mut columns = product.output_columns;
            if product.inner_block == 0 || columns == 0 {
                return None;
            }
            if let Some(group) = out.format.layout.order.gemm_output_group() {
                columns = columns.min(group);
            }
            let calls = u64::from(inputs[0].shape.0[left_axis].div_ceil(product.inner_block))
                .checked_mul(u64::from(out.shape.0[column_axis].div_ceil(columns)))?;
            let inner = product.inner_block.min(inputs[0].shape.0[left_axis]);
            columns = columns.min(out.shape.0[column_axis]);
            inputs[0].shape.0[left_axis] = inner;
            inputs[1].shape.0[right_axis] = inner;
            out.shape.0[column_axis] = columns;
            price.total = super::primitive::kernel_cycles(
                &TileKernelSpec::Gemm {
                    multiply: product.multiply,
                    accumulate: product.accumulate,
                    mode: product.mode,
                    weights: crate::GemmWeightLoad::Standard,
                    inner_block: inner,
                    output_columns: columns,
                },
                |i| inputs.get(i).map(super::primitive::Geometry::Tensor),
                super::primitive::Geometry::Tensor(&out),
            )
            .saturating_mul(calls);
        }
        MidOperationKind::Compute(Compute::Sum { axis, staging }) => {
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
        MidOperationKind::Copy { policy, .. } => {
            let input = tensor(operation.inputs[0]);
            let bytes = maximum_shard_bytes(output);
            let local_conversion = *policy == crate::CopyPolicy::LocalKernel;
            let same_ownership = local_conversion
                || (crate::tensor::same_distribution(input, output)
                    && values[operation.inputs[0].index() as usize].owners
                        == values[operation.results[0].index() as usize].owners);
            if !same_ownership
                || matches!(
                    operation.kind,
                    MidOperationKind::Copy {
                        mapping: crate::CoordinateMapping { view: Some(_), .. },
                        ..
                    }
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
                    MidOperationKind::Copy { mapping, .. }
                    if !mapping.is_identity());
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
        }
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

pub(crate) fn region_program(
    tile_count: u16,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MidProgram {
    crate::MidProgram {
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
        outputs: outputs.to_vec(),
        ..crate::MidProgram::default()
    }
}
