//! Coarse costing of whole-device primitives. Work scales with the selected
//! tensor program and axis partitions, not the number of tile IR objects.

use super::*;
use crate::mid::MidOperationKind;
use ipu_target::Target;

use crate::MidGraph;

/// Price complete replacement sequences with the same overflow and missing-cost rules.
pub(crate) fn operation_cycles<'a>(
    target: Target,
    operations: impl IntoIterator<Item = &'a MidOperation>,
    values: &[MidValue],
    tile_count: u16,
) -> Option<u64> {
    operations.into_iter().try_fold(0u64, |sum, op| {
        operation_cost(target, op, values, tile_count)
            .map(|(cost, _, _)| sum.saturating_add(cost.total))
    })
}

pub(crate) fn analyze(
    target: Target,
    program: &MidGraph,
    copies: &BTreeMap<MidValueId, u32>,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    analyze_observed(target, program, copies, &mut ())
}

/// The normal estimator uses the zero-cost observer. Diagnostics record the
/// same allocation and liveness decisions, without a second accounting model.
pub(crate) trait MemoryObserver {
    const SPLIT_RESIDENT: bool = false;
    fn resident(&mut self, _origin: crate::ValueId, _class: MemoryClass, _bytes: &[u64]) {}
    fn nonresident(&mut self, _usage: &[MemoryUsage], _maximum_standard: u64) {}
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

pub(crate) fn analyze_observed(
    target: Target,
    program: &MidGraph,
    copies: &BTreeMap<MidValueId, u32>,
    observer: &mut impl MemoryObserver,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    analyze_storage::<true, _>(target, program, copies, observer)
}

/// A fitting upper bound needs no refinement. Every failed capacity screen is
/// checked on actual owners before it can discard a candidate.
pub(crate) fn analyze_with_budget(
    program: &MidGraph,
    copies: &BTreeMap<MidValueId, u32>,
    config: &crate::PipelineConfig,
) -> Option<(ProgramCycles, MemoryPeaks)> {
    let bound = analyze_storage::<false, _>(config.target, program, copies, &mut ())?;
    if bound.1.fits_with_budget(
        config.target,
        config.standard_memory_reservation_bytes,
        config.tile_memory_budget_bytes,
    ) {
        Some(bound)
    } else {
        analyze(config.target, program, copies)
    }
}

fn analyze_storage<const PER_TILE: bool, O: MemoryObserver>(
    target: Target,
    program: &MidGraph,
    copies: &BTreeMap<MidValueId, u32>,
    observer: &mut O,
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
        for &(output, input) in operation.output_aliases.as_slice() {
            alias(operation.results[output], operation.inputs[input]);
        }
        if let MidOperationKind::Repeat(repeat) = &operation.kind {
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
    }
    for id in 0..parent.len() {
        parent[id] = root(&parent, id);
    }
    let roots = parent;
    let mut element = vec![false; roots.len()];
    let mut tail = vec![0; roots.len()];
    for (operation, _) in &steps {
        if let MidOperationKind::Gemm { multiply, .. } = &operation.kind {
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
                target.interleaved_memory_element_bytes()
            } else {
                target.standard_memory_element_bytes()
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
            let tile = if PER_TILE {
                usize::from(value.owners.tile(owner, program.tile_count)?)
            } else {
                0
            };
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
    let mut resident = vec![false; if O::SPLIT_RESIDENT { roots.len() } else { 0 }];
    let mut tile_resident = vec![MemoryUsage::default(); if O::SPLIT_RESIDENT { tiles } else { 0 }];
    for input in &program.inputs {
        if !O::SPLIT_RESIDENT || input.kind != crate::GraphInputKind::Parameter {
            continue;
        }
        let id = roots[input.value.index() as usize];
        if !resident[id] {
            observer.resident(
                program.values[input.value.index() as usize].origin,
                classes[id],
                &bytes[id],
            );
            for (usage, &size) in tile_resident.iter_mut().zip(&bytes[id]) {
                usage.add_class(classes[id], size);
            }
            resident[id] = true;
        }
    }
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
    let observe_nonresident = |observer: &mut _,
                               usage: &[MemoryUsage],
                               live: &[bool],
                               scratch: u64| {
        if !O::SPLIT_RESIDENT {
            return;
        }
        let usage = usage
            .iter()
            .zip(&tile_resident)
            .map(|(all, resident)| MemoryUsage {
                standard: all.standard - resident.standard,
                interleaved: all.interleaved - resident.interleaved,
            })
            .collect::<Vec<_>>();
        let maximum = (0..live.len())
            .filter(|&id| live[id] && !resident[id] && classes[id] == MemoryClass::Ipu21Standard)
            .map(|id| maximum_sizes[id])
            .max()
            .unwrap_or(0)
            .max(scratch);
        MemoryObserver::nonresident(observer, &usage, maximum);
    };
    observe_nonresident(observer, &tile_live, &live, 0);

    for (index, (operation, count)) in steps.into_iter().enumerate() {
        for value in operation.inputs.iter().chain(&operation.results) {
            live[roots[value.index() as usize]] = true;
        }
        let (price, scratch, row_bytes) =
            operation_cost(target, operation, &program.values, program.tile_count)?;
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
        observe_nonresident(observer, &tile_usage, &live, scratch.standard);
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

pub(crate) fn operation_cost(
    target: Target,
    operation: &MidOperation,
    values: &[MidValue],
    tile_count: u16,
) -> Option<(ProgramCycles, MemoryUsage, u64)> {
    let tensor = |id: MidValueId| &values[id.index() as usize].tensor_type;
    if matches!(operation.kind, MidOperationKind::Repeat(_)) {
        return Some((ProgramCycles::default(), MemoryUsage::default(), 0));
    }
    let output = tensor(*operation.results.first()?);
    let outputs = operation
        .results
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            operation
                .output_windows
                .get(index)
                .unwrap_or(&crate::OperandWindow::default())
                .local_extents(tensor(id), true)
        })
        .collect::<Option<Vec<_>>>()?;
    let out = &outputs[0];
    let mut scratch = MemoryUsage::default();
    let mut rows = 0;
    let mut price = ProgramCycles::default();
    match &operation.kind {
        MidOperationKind::Copy {
            policy,
            packing,
            mapping,
        } => {
            // Relay scratch and its two transfer legs require concrete geometry.
            // Do not rank this explicit strategy with the direct-copy estimate.
            if *policy == crate::CopyPolicy::GatherThenMulticast {
                return None;
            }
            let input = tensor(operation.inputs[0]);
            let bytes = maximum_shard_bytes(output);
            let policy = if *policy == crate::CopyPolicy::Automatic {
                crate::default_copy_policy(&input.format.layout, &output.format.layout)
            } else {
                *policy
            };
            let traffic = conversion_traffic(
                target,
                &values[operation.inputs[0].index() as usize],
                &values[operation.results[0].index() as usize],
                mapping,
                tile_count,
            );
            let (local_bytes, local_calls) = if let Some(traffic) = &traffic {
                if !traffic.exchange.is_empty() {
                    if policy == crate::CopyPolicy::LocalKernel {
                        return None;
                    }
                    price.exchange = super::cycles::exchange_work_cycles(
                        target,
                        traffic.paired_payload_bytes,
                        traffic.exchange.maximum_control_cycles(target),
                    )
                    .saturating_add(target.costs().exchange_phase_cycles);
                    let fragments = if mapping.is_identity() {
                        super::movement::grid_fragments(input, output).unwrap_or(0)
                    } else {
                        0
                    }
                    .max(traffic.exchange.maximum_fragments());
                    let (fragment_cycles, footprint) =
                        exchange_fragment_price(target, traffic.paired_payload_bytes, 1, fragments);
                    price.exchange = price.exchange.max(fragment_cycles);
                    rows = footprint;
                }
                if policy == crate::CopyPolicy::DirectRetile {
                    (
                        traffic.maximum_local_bytes,
                        traffic.maximum_local_intersections,
                    )
                } else {
                    (
                        traffic.maximum_destination_bytes.saturating_mul(2),
                        traffic.maximum_intersections,
                    )
                }
            } else {
                (bytes, 1)
            };
            let local_conversion = policy == crate::CopyPolicy::LocalKernel;
            let same_ownership = local_conversion
                || (crate::tensor::same_distribution(input, output)
                    && values[operation.inputs[0].index() as usize].owners
                        == values[operation.results[0].index() as usize].owners);
            if traffic.is_none() && (!same_ownership || mapping.view.is_some()) {
                let destinations = u64::from(output.format.layout.tiling.tile_count);
                let sources = u64::from(input.format.layout.tiling.tile_count).max(1);
                let sends = bytes
                    .saturating_mul(destinations)
                    .div_ceil(sources)
                    .min(maximum_shard_bytes(input));
                let payload = bytes.max(sends);
                let fragments = mapping
                    .is_identity()
                    .then(|| super::movement::grid_fragments(input, output))
                    .flatten()
                    .unwrap_or_else(|| {
                        payload.div_ceil(movement_fragment_bytes(input, output).max(1))
                    });
                let (exchange, footprint) = exchange_fragment_price(target, payload, 1, fragments);
                price.exchange = exchange;
                rows = footprint;
            }
            price.total = price
                .exchange
                .saturating_add(local_bytes.div_ceil(target.costs().local_copy_bytes_per_cycle))
                .saturating_add(local_calls.saturating_mul(target.costs().local_copy_call_cycles));
            if *packing == crate::PackingPolicy::Staged
                && input.format.precision == output.format.precision
                && input.format.layout.order != output.format.layout.order
            {
                scratch.standard = bytes;
                // Packed destinations are populated from row-major staging.
                // Other mappings use generic copies, already charged above;
                // absence of a specialized kernel does not make them invalid.
                let from = if output.format.layout.order == ElementOrder::RowMajor {
                    input.format.layout.order
                } else {
                    ElementOrder::RowMajor
                };
                if crate::kernel::rearrange::supported(
                    from,
                    output.format.layout.order,
                    output.format.precision,
                ) {
                    price.total = price
                        .total
                        .saturating_add(crate::kernel::rearrange::estimate(
                            target,
                            from,
                            crate::storage::TensorStorage {
                                format: &output.format,
                                extents: out,
                            },
                            crate::storage::TensorStorage {
                                format: &output.format,
                                extents: out,
                            },
                        ));
                }
            }
        }
        MidOperationKind::Repeat(_) => unreachable!(),
        kernel => {
            let inputs = operation
                .inputs
                .iter()
                .zip(&operation.operands)
                .map(|(&id, indexing)| match indexing {
                    crate::OperandIndexing::Local(window) => {
                        window.local_extents(tensor(id), false)
                    }
                    crate::OperandIndexing::Fragment(window) => {
                        window.local_extents(tensor(id), true)
                    }
                    crate::OperandIndexing::Elementwise { .. } => {
                        crate::OperandWindow::default().local_extents(tensor(id), false)
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            let inputs = inputs
                .iter()
                .zip(&operation.inputs)
                .map(|(extents, &id)| crate::storage::TensorStorage {
                    format: &tensor(id).format,
                    extents,
                })
                .collect::<Vec<_>>();
            let outputs = outputs
                .iter()
                .zip(&operation.results)
                .map(|(extents, &id)| crate::storage::TensorStorage {
                    format: &tensor(id).format,
                    extents,
                })
                .collect::<Vec<_>>();
            price.total = if matches!(kernel, MidOperationKind::Gemm { .. }) {
                crate::kernel::gemm::invocations(
                    kernel,
                    [inputs[0].extents, inputs[1].extents, outputs[0].extents],
                )
                .map_or(u64::MAX, |calls| {
                    calls.into_iter().fold(0u64, |cost, (kind, regions)| {
                        let input = [
                            crate::storage::TensorStorage {
                                format: inputs[0].format,
                                extents: &regions[0],
                            },
                            crate::storage::TensorStorage {
                                format: inputs[1].format,
                                extents: &regions[1],
                            },
                        ];
                        let output = [crate::storage::TensorStorage {
                            format: outputs[0].format,
                            extents: &regions[2],
                        }];
                        cost.saturating_add(
                            crate::kernel::KernelCall::select(target, &kind, &input, &output, None)
                                .map_or(u64::MAX, |call| call.cycles),
                        )
                    })
                })
            } else {
                crate::kernel::KernelCall::select(target, kernel, &inputs, &outputs, None)
                    .map_or(u64::MAX, |call| call.cycles)
            };
        }
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

pub(crate) fn region_program(
    tile_count: u16,
    initial: &[MidValueId],
    operations: &[MidOperation],
    outputs: &[MidValueId],
    values: &[MidValue],
) -> MidGraph {
    crate::MidGraph {
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
        ..crate::MidGraph::default()
    }
}
