use anyhow::{Context, Result, bail};
use ipu_codegen::{
    AttentionScale, CompiledPackage, CompiledTensor, CompiledTensorShard, ComputeGraph,
    GemmOptions, Operation, OperationKind, Precision, Region, Repeat, ShardExtent, ShardView,
    ValueId, logical_view_byte_spans,
};
use ipu_driver::{Device, DriverError, TileException};
use ipu_package::{Application, Binding};
use ipu_runtime::Runtime;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

#[derive(Clone, Debug)]
pub(crate) struct HostTensor {
    pub(crate) shape: Vec<u32>,
    pub(crate) values: Vec<f32>,
}

#[derive(Clone, Copy)]
struct ComparisonOptions {
    samples: usize,
    atol: f32,
    rtol: f32,
}

pub(crate) type PreparedInputs = (BTreeMap<ValueId, HostTensor>, Vec<u8>, Vec<u8>);

pub fn run(
    runtime: &Runtime,
    graph: &ComputeGraph,
    package: &CompiledPackage,
    samples: usize,
    atol: f32,
    rtol: f32,
    timeout: Duration,
) -> Result<()> {
    let (values, weights, inputs) =
        prepare_inputs(graph, &package.application, &package.tensors.inputs)?;
    let references = evaluate(graph, values, &package.tensors.precisions)?;
    let mut session = runtime.host_session(&package.application)?;
    session.start()?;
    if !package.application.weights.is_empty() {
        let initialized = session.invoke_streaming_deferred("initialize", &weights)?;
        session.collect(&initialized)?;
    }

    let mut next = 0usize;
    let mut waiting_for_resume = false;
    let comparison = ComparisonOptions {
        samples,
        atol,
        rtol,
    };
    let executed = session
        .invoke_streaming_deferred_with_poll("run", &inputs, |device| {
            service_checkpoint(
                device,
                &package.application,
                package,
                &references,
                &mut next,
                &mut waiting_for_resume,
                comparison,
            )
            .map_err(|error| DriverError::Invalid(error.to_string()))
        })
        .inspect_err(|error| {
            tracing::error!(
                %error,
                completed = next,
                device = %super::device_failure_diagnostics(runtime, &package.application),
                "diagnostic run failed"
            );
        })?;
    if next != package.checkpoints.len() {
        bail!(
            "program completed after {next} diagnostic checkpoints; expected {}",
            package.checkpoints.len()
        );
    }
    runtime
        .device()
        .write_sync_mark(ipu_driver::pci::HSP_GS2_CONTROL, 1)?;
    super::diagnose_completion(runtime, &package.application, timeout)?;
    let _ = session.collect(&executed)?;
    println!(
        "diagnosticCheckpoints={} sampleLimit={} numericalTest=PASS",
        next, samples
    );
    Ok(())
}

fn service_checkpoint(
    device: &Device,
    application: &Application,
    package: &CompiledPackage,
    references: &BTreeMap<ValueId, HostTensor>,
    next: &mut usize,
    waiting_for_resume: &mut bool,
    comparison: ComparisonOptions,
) -> Result<()> {
    let Some(first) = application.tiles.first() else {
        bail!("diagnostic application has no tiles");
    };
    let first_stopped = device.tile_context_state(u16::try_from(first.physical_tile)?, 0)? == 2;
    if *waiting_for_resume {
        if first_stopped {
            let status = device.read_tile_context_status(u16::try_from(first.physical_tile)?, 0)?;
            let previous = package
                .checkpoints
                .get(next.saturating_sub(1))
                .context("previous diagnostic checkpoint is missing")?;
            let previous_exception = if previous.breakpoint == 0 {
                TileException::PatchedBreak0
            } else {
                TileException::PatchedBreak1
            };
            if TileException::from_status(status) == previous_exception {
                return Ok(());
            }
        }
        *waiting_for_resume = false;
    }
    if !first_stopped {
        return Ok(());
    }
    for tile in &application.tiles {
        if device.tile_context_state(u16::try_from(tile.physical_tile)?, 0)? != 2 {
            return Ok(());
        }
    }
    let checkpoint = package
        .checkpoints
        .get(*next)
        .with_context(|| format!("unexpected extra device checkpoint {}", *next))?;
    let expected_exception = if checkpoint.breakpoint == 0 {
        TileException::PatchedBreak0
    } else {
        TileException::PatchedBreak1
    };
    for tile in &application.tiles {
        let physical = u16::try_from(tile.physical_tile)?;
        let status = device.read_tile_context_status(physical, 0)?;
        let exception = TileException::from_status(status);
        if exception != expected_exception {
            bail!(
                "tile {physical} stopped at operator {} with {exception} (status {status:#x})",
                checkpoint.operation.index()
            );
        }
    }
    let mut checked = 0usize;
    let mut maximum_error = 0.0f32;
    for tensor in &checkpoint.tensors {
        let expected = references.get(&tensor.value).with_context(|| {
            format!(
                "host reference for value {} is missing",
                tensor.value.index()
            )
        })?;
        if tensor.shards.is_empty() {
            tracing::warn!(
                operation = checkpoint.operation.index(),
                value = tensor.value.index(),
                "operator result has no materialized canonical storage; skipping readback"
            );
            continue;
        }
        let (tensor_checked, tensor_error) = compare_tensor(
            device,
            tensor,
            expected,
            comparison.samples,
            comparison.atol,
            comparison.rtol,
            checkpoint.operation.index(),
        )?;
        checked += tensor_checked;
        maximum_error = maximum_error.max(tensor_error);
    }
    println!(
        "checkpoint={} operation={} tensors={} samples={} maximumAbsoluteError={maximum_error:.6}",
        *next,
        checkpoint.operation.index(),
        checkpoint.tensors.len(),
        checked
    );
    // PBRK records the trap's own PC. Step over this dedicated checkpoint like
    // a debugger does for a software breakpoint, then clear the exception.
    const IPU21_NOP_INSTRUCTION: u32 = 0x19e0_0000;
    for tile in &application.tiles {
        let physical = u16::try_from(tile.physical_tile)?;
        let pc = device.read_tile_program_counter(physical, 0)?;
        device.write_tile_word_from_stopped_context(physical, 0, pc, IPU21_NOP_INSTRUCTION)?;
        device.clear_tile_exception(physical, 0)?;
    }
    *waiting_for_resume = true;
    *next += 1;
    Ok(())
}

fn compare_tensor(
    device: &Device,
    tensor: &CompiledTensor,
    expected: &HostTensor,
    sample_limit: usize,
    atol: f32,
    rtol: f32,
    operation: u32,
) -> Result<(usize, f32)> {
    let total = usize::try_from(tensor.shape.elements())?;
    let wanted = sample_indices(total, sample_limit);
    let mut actual = BTreeMap::<usize, f32>::new();
    let mut words = HashMap::<(u16, u32), u32>::new();
    for shard in &tensor.shards {
        for (index, byte_offset) in shard_elements(tensor, shard)? {
            if !wanted.contains(&index) || actual.contains_key(&index) {
                continue;
            }
            let address = shard
                .address
                .checked_add(byte_offset)
                .context("diagnostic tensor address overflow")?;
            let word_address = address & !0b11;
            let word = *words
                .entry((shard.physical_tile, word_address))
                .or_insert(device.read_tile_word(shard.physical_tile, word_address)?);
            actual.insert(index, decode_word(word, address & 0b11, tensor.precision)?);
        }
    }
    let mut mismatches = Vec::new();
    let mut maximum = 0.0f32;
    for &index in &wanted {
        let observed = actual.get(&index).with_context(|| {
            format!(
                "value {} sample {index} is absent from all diagnostic shards",
                tensor.value.index()
            )
        })?;
        let reference = *expected
            .values
            .get(index)
            .context("host reference sample is out of range")?;
        let error = (*observed - reference).abs();
        maximum = maximum.max(error);
        if (!observed.is_finite() || error > atol + rtol * reference.abs()) && mismatches.len() < 16
        {
            mismatches.push((index, reference, *observed, error));
        }
    }
    if !mismatches.is_empty() {
        bail!(
            "operator {operation} value {} numerical comparison failed: {mismatches:?}",
            tensor.value.index()
        );
    }
    Ok((wanted.len(), maximum))
}

fn sample_indices(total: usize, limit: usize) -> BTreeSet<usize> {
    let count = total.min(limit.max(1));
    if count == total {
        return (0..total).collect();
    }
    (0..count)
        .map(|index| index * (total - 1) / (count - 1).max(1))
        .collect()
}

pub(crate) fn shard_elements(
    tensor: &CompiledTensor,
    shard: &CompiledTensorShard,
) -> Result<Vec<(usize, u32)>> {
    let logical_extents = shard
        .storage
        .extents
        .iter()
        .map(|extent| ShardExtent {
            physical_end: extent.logical_end,
            ..*extent
        })
        .collect::<Vec<_>>();
    let view = ShardView {
        shard: shard.storage.id,
        extents: logical_extents.clone().into(),
    };
    let element_bytes = u32::try_from(tensor.precision.bytes())?;
    let offsets = logical_view_byte_spans(&shard.storage, &view)?
        .into_iter()
        .flat_map(|span| (span.offset..span.offset + span.bytes).step_by(element_bytes as usize))
        .collect::<Vec<_>>();
    let widths = logical_extents
        .iter()
        .map(|extent| extent.logical_end - extent.start)
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(offsets.len());
    for (local, offset) in offsets.into_iter().enumerate() {
        let mut remainder = u64::try_from(local)?;
        let mut global = 0u64;
        let mut stride = 1u64;
        for (extent, (&width, &dimension)) in logical_extents
            .iter()
            .zip(widths.iter().zip(&tensor.shape.0))
            .rev()
        {
            let coordinate = remainder % u64::from(width);
            remainder /= u64::from(width);
            global += (u64::from(extent.start) + coordinate) * stride;
            stride *= u64::from(dimension);
        }
        result.push((usize::try_from(global)?, offset));
    }
    Ok(result)
}

fn decode_word(word: u32, byte: u32, precision: Precision) -> Result<f32> {
    Ok(match precision {
        Precision::F16 => super::half_to_f32(((word >> (byte * 8)) & 0xffff) as u16),
        Precision::F32 if byte == 0 => f32::from_bits(word),
        Precision::F32 => bail!("unaligned F32 diagnostic value"),
        Precision::F8F143 { .. } => bail!("F8 diagnostic decoding is not implemented"),
    })
}

pub(crate) fn prepare_inputs(
    graph: &ComputeGraph,
    application: &Application,
    metadata: &[CompiledTensor],
) -> Result<PreparedInputs> {
    let mut values = BTreeMap::new();
    for input in graph.inputs() {
        let metadata = metadata
            .iter()
            .find(|tensor| tensor.value == input.value)
            .with_context(|| format!("diagnostic metadata for {} is missing", input.name))?;
        let scale = match input.kind {
            ipu_codegen::GraphInputKind::Host => 0.25,
            ipu_codegen::GraphInputKind::Parameter => 0.0625,
        };
        let seed = 0x4449_4147_4e4f_5354 ^ u64::from(input.value.index());
        let data = (0..input.shape.elements())
            .map(|index| quantize(super::gaussian(seed, index) * scale, metadata.precision))
            .collect();
        values.insert(
            input.value,
            HostTensor {
                shape: input.shape.0.clone(),
                values: data,
            },
        );
    }
    let pack = |bindings: &[Binding]| -> Result<Vec<u8>> {
        let mut result = Vec::new();
        for binding in bindings {
            let metadata = metadata
                .iter()
                .find(|tensor| tensor.name.as_deref() == Some(binding.name.as_str()))
                .with_context(|| format!("diagnostic metadata for {} is missing", binding.name))?;
            let tensor = &values[&metadata.value];
            let mut bytes = vec![0; usize::try_from(super::binding_size(binding))?];
            let mut covered = vec![false; usize::try_from(metadata.shape.elements())?];
            for shard in &metadata.shards {
                let slice = binding
                    .slices
                    .iter()
                    .find(|slice| {
                        slice.tile == u32::from(shard.physical_tile)
                            && slice.tile_address == shard.address
                    })
                    .with_context(|| {
                        format!("binding slice for {} shard is missing", binding.name)
                    })?;
                for (index, offset) in shard_elements(metadata, shard)? {
                    if u64::from(offset) + u64::try_from(metadata.precision.bytes())? > slice.size {
                        bail!("logical element exceeds binding {} shard", binding.name);
                    }
                    encode_value(
                        &mut bytes,
                        usize::try_from(slice.file_offset + u64::from(offset))?,
                        tensor.values[index],
                        metadata.precision,
                    )?;
                    covered[index] = true;
                }
            }
            if let Some(missing) = covered.iter().position(|covered| !covered) {
                bail!(
                    "binding {} does not store logical element {missing}",
                    binding.name
                );
            }
            result.extend(bytes);
        }
        Ok(result)
    };
    let weights = pack(&application.weights)?;
    let inputs = pack(&application.inputs)?;
    Ok((values, weights, inputs))
}

fn encode_value(bytes: &mut [u8], offset: usize, value: f32, precision: Precision) -> Result<()> {
    match precision {
        Precision::F16 => {
            bytes[offset..offset + 2].copy_from_slice(&super::f32_to_half(value).to_le_bytes())
        }
        Precision::F32 => bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes()),
        Precision::F8F143 { .. } => bail!("F8 diagnostic encoding is not implemented"),
    }
    Ok(())
}

pub(crate) fn evaluate(
    graph: &ComputeGraph,
    mut values: BTreeMap<ValueId, HostTensor>,
    precisions: &BTreeMap<ValueId, Precision>,
) -> Result<BTreeMap<ValueId, HostTensor>> {
    evaluate_operations(graph.operations(), graph, &mut values, precisions)?;
    Ok(values)
}

fn evaluate_operations(
    operations: &[Operation],
    graph: &ComputeGraph,
    values: &mut BTreeMap<ValueId, HostTensor>,
    precisions: &BTreeMap<ValueId, Precision>,
) -> Result<()> {
    for operation in operations {
        let results = match &operation.kind {
            OperationKind::Gemm(options) => vec![gemm(
                &values[&operation.inputs[0]],
                &values[&operation.inputs[1]],
                *options,
            )?],
            OperationKind::Gelu => vec![map_unary(
                &values[&operation.inputs[0]],
                super::gelu_reference,
            )],
            OperationKind::Add(_) => vec![add(
                &values[&operation.inputs[0]],
                &values[&operation.inputs[1]],
            )?],
            OperationKind::SplitHeads(options) => {
                vec![split_heads(&values[&operation.inputs[0]], options.heads)?]
            }
            OperationKind::FlashAttention(options) => vec![attention(
                &values[&operation.inputs[0]],
                &values[&operation.inputs[1]],
                &values[&operation.inputs[2]],
                *options,
            )?],
            OperationKind::Repeat(repeat) => {
                repeat_values(operation, repeat, graph, values, precisions)?
            }
        };
        for (&id, mut tensor) in operation.results.iter().zip(results) {
            if let Some(&precision) = precisions.get(&id) {
                for value in &mut tensor.values {
                    *value = quantize(*value, precision);
                }
            }
            values.insert(id, tensor);
        }
    }
    Ok(())
}

fn repeat_values(
    operation: &Operation,
    repeat: &Repeat,
    graph: &ComputeGraph,
    values: &BTreeMap<ValueId, HostTensor>,
    precisions: &BTreeMap<ValueId, Precision>,
) -> Result<Vec<HostTensor>> {
    let mut carried = operation.inputs[..repeat.carried_inputs]
        .iter()
        .map(|id| values[id].clone())
        .collect::<Vec<_>>();
    let invariants = operation.inputs
        [repeat.carried_inputs..repeat.carried_inputs + repeat.invariant_inputs]
        .iter()
        .map(|id| values[id].clone())
        .collect::<Vec<_>>();
    for iteration in 0..repeat.count as usize {
        let mut local = values.clone();
        for (argument, value) in repeat
            .body
            .arguments
            .iter()
            .take(repeat.carried_inputs)
            .zip(&carried)
        {
            local.insert(*argument, value.clone());
        }
        for (argument, value) in repeat
            .body
            .arguments
            .iter()
            .skip(repeat.carried_inputs)
            .take(repeat.invariant_inputs)
            .zip(&invariants)
        {
            local.insert(*argument, value.clone());
        }
        for (argument, sequence) in repeat
            .body
            .arguments
            .iter()
            .skip(repeat.carried_inputs + repeat.invariant_inputs)
            .zip(&repeat.iterated_inputs)
        {
            let sequence = &graph.sequences()[sequence.index() as usize];
            local.insert(*argument, values[&sequence.values[iteration]].clone());
        }
        evaluate_region(&repeat.body, graph, &mut local, precisions)?;
        carried = repeat
            .body
            .yields
            .iter()
            .map(|id| local[id].clone())
            .collect();
    }
    Ok(carried)
}

fn evaluate_region(
    region: &Region,
    graph: &ComputeGraph,
    values: &mut BTreeMap<ValueId, HostTensor>,
    precisions: &BTreeMap<ValueId, Precision>,
) -> Result<()> {
    evaluate_operations(&region.operations, graph, values, precisions)
}

fn gemm(left: &HostTensor, right: &HostTensor, options: GemmOptions) -> Result<HostTensor> {
    if left.shape.len() < 2 || right.shape.len() < 2 {
        bail!("GEMM diagnostic operands must have rank at least two");
    }
    let (lm, lk) = matrix_dims(&left.shape, options.transpose_left);
    let (rk, rn) = matrix_dims(&right.shape, options.transpose_right);
    if lk != rk {
        bail!("GEMM diagnostic inner dimensions differ");
    }
    let batch_shape = broadcast_shape(
        &left.shape[..left.shape.len() - 2],
        &right.shape[..right.shape.len() - 2],
    )?;
    let batches = product(&batch_shape);
    let mut output = vec![0.0; batches * lm as usize * rn as usize];
    let left_matrix_elements = product(&left.shape[left.shape.len() - 2..]);
    let right_matrix_elements = product(&right.shape[right.shape.len() - 2..]);
    let left_stride = *left.shape.last().context("left GEMM shape is empty")?;
    let right_stride = *right.shape.last().context("right GEMM shape is empty")?;
    for batch in 0..batches {
        let coords = decode_index(batch, &batch_shape);
        let lb = broadcast_batch_offset(&coords, &batch_shape, &left.shape[..left.shape.len() - 2]);
        let rb =
            broadcast_batch_offset(&coords, &batch_shape, &right.shape[..right.shape.len() - 2]);
        let left = &left.values[lb * left_matrix_elements..][..left_matrix_elements];
        let right = &right.values[rb * right_matrix_elements..][..right_matrix_elements];
        let output = &mut output[batch * lm as usize * rn as usize..][..lm as usize * rn as usize];
        host_sgemm(
            left,
            right,
            output,
            lm,
            rn,
            lk,
            left_stride,
            right_stride,
            options.transpose_left,
            options.transpose_right,
        )?;
    }
    let mut shape = batch_shape;
    shape.extend([lm, rn]);
    Ok(HostTensor {
        shape,
        values: output,
    })
}

#[link(name = "openblas")]
unsafe extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transpose_a: i32,
        transpose_b: i32,
        rows: i32,
        columns: i32,
        inner: i32,
        alpha: f32,
        left: *const f32,
        left_stride: i32,
        right: *const f32,
        right_stride: i32,
        beta: f32,
        output: *mut f32,
        output_stride: i32,
    );
}

#[allow(clippy::too_many_arguments)]
fn host_sgemm(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    rows: u32,
    columns: u32,
    inner: u32,
    left_stride: u32,
    right_stride: u32,
    transpose_left: bool,
    transpose_right: bool,
) -> Result<()> {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANSPOSE: i32 = 111;
    const CBLAS_TRANSPOSE: i32 = 112;
    let rows = i32::try_from(rows)?;
    let columns = i32::try_from(columns)?;
    let inner = i32::try_from(inner)?;
    let left_stride = i32::try_from(left_stride)?;
    let right_stride = i32::try_from(right_stride)?;
    if output.len() != usize::try_from(rows)? * usize::try_from(columns)? {
        bail!("host GEMM output dimensions are inconsistent");
    }
    // SAFETY: callers provide complete row-major matrix slices, the leading
    // dimensions come from their physical shapes, and output is exclusive.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            if transpose_left {
                CBLAS_TRANSPOSE
            } else {
                CBLAS_NO_TRANSPOSE
            },
            if transpose_right {
                CBLAS_TRANSPOSE
            } else {
                CBLAS_NO_TRANSPOSE
            },
            rows,
            columns,
            inner,
            1.0,
            left.as_ptr(),
            left_stride,
            right.as_ptr(),
            right_stride,
            0.0,
            output.as_mut_ptr(),
            columns,
        );
    }
    Ok(())
}

fn matrix_dims(shape: &[u32], transpose: bool) -> (u32, u32) {
    let pair = (shape[shape.len() - 2], shape[shape.len() - 1]);
    if transpose { (pair.1, pair.0) } else { pair }
}

#[cfg(test)]
fn matrix_index(shape: &[u32], batch: usize, row: u32, column: u32, transpose: bool) -> usize {
    let rows = shape[shape.len() - 2] as usize;
    let columns = shape[shape.len() - 1] as usize;
    let (row, column) = if transpose {
        (column, row)
    } else {
        (row, column)
    };
    batch * rows * columns + row as usize * columns + column as usize
}

fn add(left: &HostTensor, right: &HostTensor) -> Result<HostTensor> {
    let shape = broadcast_shape(&left.shape, &right.shape)?;
    let values = (0..product(&shape))
        .map(|index| {
            let coords = decode_index(index, &shape);
            left.values[broadcast_offset(&coords, &shape, &left.shape)]
                + right.values[broadcast_offset(&coords, &shape, &right.shape)]
        })
        .collect();
    Ok(HostTensor { shape, values })
}

fn split_heads(input: &HostTensor, heads: u32) -> Result<HostTensor> {
    let [batch, rows, width] = input.shape.as_slice() else {
        bail!("SplitHeads diagnostic input must have rank three");
    };
    if !width.is_multiple_of(heads) {
        bail!("SplitHeads diagnostic width is not divisible by heads");
    }
    let channels = width / heads;
    let mut values = vec![0.0; input.values.len()];
    for b in 0..*batch {
        for h in 0..heads {
            for r in 0..*rows {
                for c in 0..channels {
                    let source = ((b * rows + r) * width + h * channels + c) as usize;
                    let target = (((b * heads + h) * rows + r) * channels + c) as usize;
                    values[target] = input.values[source];
                }
            }
        }
    }
    Ok(HostTensor {
        shape: vec![batch * heads, *rows, channels],
        values,
    })
}

fn attention(
    query: &HostTensor,
    key: &HostTensor,
    value: &HostTensor,
    options: ipu_codegen::AttentionOptions,
) -> Result<HostTensor> {
    let [streams, query_rows, width] = query.shape.as_slice() else {
        bail!("attention query must have rank three")
    };
    let [key_streams, key_rows, key_width] = key.shape.as_slice() else {
        bail!("attention key must have rank three")
    };
    let [value_streams, value_rows, value_width] = value.shape.as_slice() else {
        bail!("attention value must have rank three")
    };
    if streams != key_streams
        || streams != value_streams
        || width != key_width
        || key_rows != value_rows
    {
        bail!("attention diagnostic shapes are incompatible");
    }
    let scale = match options.scale {
        AttentionScale::InverseSqrtQueryWidth => 1.0 / (*width as f32).sqrt(),
        AttentionScale::ValueBits(bits) => f32::from_bits(bits),
    };
    let mut output = vec![0.0; (*streams * *query_rows * *value_width) as usize];
    let mut logits = vec![0.0; *key_rows as usize];
    for stream in 0..*streams {
        for row in 0..*query_rows {
            let allowed = if options.causal {
                (row + 1).min(*key_rows)
            } else {
                *key_rows
            };
            for column in 0..allowed {
                let mut dot = 0.0;
                for inner in 0..*width {
                    dot += query.values[((stream * query_rows + row) * width + inner) as usize]
                        * key.values[((stream * key_rows + column) * width + inner) as usize];
                }
                logits[column as usize] = dot * scale;
            }
            let maximum = logits[..allowed as usize]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = logits[..allowed as usize]
                .iter_mut()
                .map(|logit| {
                    *logit = (*logit - maximum).exp();
                    *logit
                })
                .sum();
            for column in 0..*value_width {
                let mut result = 0.0;
                for inner in 0..allowed {
                    result += logits[inner as usize] / sum
                        * value.values
                            [((stream * key_rows + inner) * value_width + column) as usize];
                }
                output[((stream * query_rows + row) * value_width + column) as usize] = result;
            }
        }
    }
    Ok(HostTensor {
        shape: vec![*streams, *query_rows, *value_width],
        values: output,
    })
}

fn map_unary(input: &HostTensor, function: impl Fn(f32) -> f32) -> HostTensor {
    HostTensor {
        shape: input.shape.clone(),
        values: input.values.iter().copied().map(function).collect(),
    }
}

fn broadcast_shape(left: &[u32], right: &[u32]) -> Result<Vec<u32>> {
    let rank = left.len().max(right.len());
    let mut result = vec![1; rank];
    for (axis, output) in result.iter_mut().enumerate() {
        let l = axis
            .checked_sub(rank - left.len())
            .map_or(1, |axis| left[axis]);
        let r = axis
            .checked_sub(rank - right.len())
            .map_or(1, |axis| right[axis]);
        if l != r && l != 1 && r != 1 {
            bail!("diagnostic broadcast shapes are incompatible")
        }
        *output = l.max(r);
    }
    Ok(result)
}

fn broadcast_offset(coords: &[u32], outer: &[u32], shape: &[u32]) -> usize {
    let aligned = outer.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .fold(0usize, |offset, (axis, &dimension)| {
            offset * dimension as usize
                + if dimension == 1 {
                    0
                } else {
                    coords[aligned + axis] as usize
                }
        })
}

fn broadcast_batch_offset(coords: &[u32], outer: &[u32], shape: &[u32]) -> usize {
    broadcast_offset(coords, outer, shape)
}

fn decode_index(mut index: usize, shape: &[u32]) -> Vec<u32> {
    let mut coordinates = vec![0; shape.len()];
    for (coordinate, &dimension) in coordinates.iter_mut().zip(shape).rev() {
        *coordinate = (index % dimension as usize) as u32;
        index /= dimension as usize;
    }
    coordinates
}

fn product(shape: &[u32]) -> usize {
    shape
        .iter()
        .fold(1usize, |value, dimension| value * *dimension as usize)
}

fn quantize(value: f32, precision: Precision) -> f32 {
    match precision {
        Precision::F16 => super::half_to_f32(super::f32_to_half(value)),
        Precision::F32 => value,
        Precision::F8F143 { .. } => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ipu_codegen::{AmpOrder, amp_matrix_coordinates};

    #[test]
    fn randomized_blas_gemm_matches_scalar_reference() -> Result<()> {
        let mut random = fastrand::Rng::with_seed(0x424c_4153_4745_4d4d);
        for _ in 0..256 {
            let rows = random.u32(1..=16);
            let inner = random.u32(1..=24);
            let columns = random.u32(1..=16);
            let transpose_left = random.bool();
            let transpose_right = random.bool();
            let left_shape = if transpose_left {
                vec![inner, rows]
            } else {
                vec![rows, inner]
            };
            let right_shape = if transpose_right {
                vec![columns, inner]
            } else {
                vec![inner, columns]
            };
            let left = (0..product(&left_shape))
                .map(|_| random.f32() - 0.5)
                .collect::<Vec<_>>();
            let right = (0..product(&right_shape))
                .map(|_| random.f32() - 0.5)
                .collect::<Vec<_>>();
            let mut accelerated = vec![0.0; rows as usize * columns as usize];
            host_sgemm(
                &left,
                &right,
                &mut accelerated,
                rows,
                columns,
                inner,
                *left_shape.last().unwrap(),
                *right_shape.last().unwrap(),
                transpose_left,
                transpose_right,
            )?;
            for row in 0..rows {
                for column in 0..columns {
                    let scalar = (0..inner)
                        .map(|inner| {
                            left[matrix_index(&left_shape, 0, row, inner, transpose_left)]
                                * right
                                    [matrix_index(&right_shape, 0, inner, column, transpose_right)]
                        })
                        .sum::<f32>();
                    let observed = accelerated[(row * columns + column) as usize];
                    assert!((observed - scalar).abs() <= 2.0e-5);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn randomized_gelu_reorder_preserves_logical_coordinates() -> Result<()> {
        let mut random = fastrand::Rng::with_seed(0x4745_4c55_5245_4f52);
        for _ in 0..100 {
            let rows = random.u32(1..65);
            let columns = random.u32(1..9) * 16;
            let words = rows * columns / 2;
            let permutation = [0, 4, 1, 5, 2, 6, 3, 7];
            for worker in 0..6 {
                let mut base = worker * 8;
                while base + 8 <= words {
                    for (source_word, destination_word) in permutation.iter().enumerate() {
                        for lane in 0..2 {
                            let source = base + source_word as u32;
                            let destination = base + destination_word;
                            let source_coordinates = amp_matrix_coordinates(
                                AmpOrder::Output,
                                Precision::F16,
                                rows,
                                columns,
                                source * 2 + lane,
                            )?;
                            let destination_coordinates = amp_matrix_coordinates(
                                AmpOrder::Left,
                                Precision::F16,
                                rows,
                                columns,
                                destination * 2 + lane,
                            )?;
                            assert_eq!(source_coordinates, destination_coordinates);
                        }
                    }
                    base += 48;
                }
            }
        }
        Ok(())
    }
}
