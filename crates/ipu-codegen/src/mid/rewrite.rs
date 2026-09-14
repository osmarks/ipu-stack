//! Shared graph queries and edit application for mid transformations.
use super::*;

/// Producers whose results have exactly one consuming operation and do not
/// escape the region. Repeated operands within that consumer count once.
pub(super) fn single_use_producers(
    operations: &[MidOperation],
    required: &[MidValueId],
) -> BTreeMap<MidValueId, usize> {
    let mut producers = BTreeMap::new();
    let mut readers = BTreeMap::new();
    for (index, op) in operations.iter().enumerate() {
        for &input in op.read_values() {
            readers
                .entry(input)
                .and_modify(|reader| {
                    if *reader != Some(index) {
                        *reader = None;
                    }
                })
                .or_insert(Some(index));
        }
        for &output in &op.results {
            producers.insert(output, index);
        }
    }
    for value in required {
        producers.remove(value);
    }
    producers.retain(|id, _| readers.get(id).is_some_and(Option::is_some));
    producers
}

/// Query the current graph when a rewrite is changing readers as it proceeds.
/// Like `single_use_producers`, count consuming operations, not operand slots.
pub(super) fn is_single_use(
    operations: &[MidOperation],
    required: &[MidValueId],
    value: MidValueId,
) -> bool {
    !required.contains(&value)
        && operations
            .iter()
            .filter(|op| op.read_values().any(|v| *v == value))
            .take(2)
            .count()
            == 1
}

/// Recognize an unfused, whole-value FP16-to-FP8 local conversion.
pub(super) fn fp8_cast(op: &MidOperation, values: &[MidValue]) -> Option<(MidValueId, MidValueId)> {
    let local = match &op.kind {
        MidOperationKind::Compute(Compute::Kernel {
            kernel:
                TileKernelSpec::Cast {
                    from: Precision::F16,
                    to: Precision::F8F143 { .. },
                },
            operands,
            output_aliases,
        }) => {
            operands.len() == 1
                && operands[0] == (OperandIndexing::Elementwise { result: 0 })
                && output_aliases.is_empty()
        }
        _ => false,
    };
    let ([input], [output]) = (op.inputs.as_slice(), op.results.as_slice()) else {
        return None;
    };
    (local
        && values[input.index() as usize].tensor_type.format.precision == Precision::F16
        && matches!(
            values[output.index() as usize].tensor_type.format.precision,
            Precision::F8F143 { .. }
        ))
    .then_some((*input, *output))
}

/// Apply index-based edits only after matching is complete.
pub(super) fn apply_edits(
    operations: &mut Vec<MidOperation>,
    removed: &BTreeSet<usize>,
    mut before: BTreeMap<usize, Vec<MidOperation>>,
) {
    if before.is_empty() {
        let mut index = 0;
        operations.retain(|_| {
            let keep = !removed.contains(&index);
            index += 1;
            keep
        });
        return;
    }
    *operations = std::mem::take(operations)
        .into_iter()
        .enumerate()
        .flat_map(|(index, op)| {
            before
                .remove(&index)
                .into_iter()
                .flatten()
                .chain((!removed.contains(&index)).then_some(op))
        })
        .collect();
}

pub(super) fn same_storage(a: &MidValue, b: &MidValue) -> bool {
    a.tile_offset == b.tile_offset
        && a.tensor_type.format.precision == b.tensor_type.format.precision
        && a.tensor_type.format.layout.order == b.tensor_type.format.layout.order
        && a.tensor_type.format.layout.memory_class == b.tensor_type.format.layout.memory_class
        && implementation::same_distribution(&a.tensor_type, &b.tensor_type)
}

/// Source of a single-result coordinate-preserving copy. Shapes, precision,
/// and placement may still differ; each rewrite checks the constraints it needs.
pub(super) fn coordinate_copy_source(op: &MidOperation) -> Option<MidValueId> {
    let ([input], [_]) = (op.inputs.as_slice(), op.results.as_slice()) else {
        return None;
    };
    match &op.kind {
        MidOperationKind::Copy {
            mapping, policy, ..
        } if mapping.is_identity() && *policy != CopyPolicy::LocalKernel => Some(*input),
        _ => None,
    }
}

/// Walk coordinate-preserving copies, optionally requiring identical storage.
/// Callers check extra readers and intervening writes before rewriting them.
pub(super) fn producer_through_copies(
    mut value: MidValueId,
    operations: &[MidOperation],
    values: &[MidValue],
    identity_only: bool,
) -> Option<(MidValueId, usize, Vec<usize>)> {
    let mut copies = vec![];
    loop {
        let index = operations.iter().rposition(|op| op.results == [value])?;
        let op = &operations[index];
        let Some(input) = coordinate_copy_source(op) else {
            return Some((value, index, copies));
        };
        if values[value.index() as usize].tensor_type.shape
            != values[input.index() as usize].tensor_type.shape
            || values[value.index() as usize].tensor_type.format.precision
                != values[input.index() as usize].tensor_type.format.precision
            || (identity_only
                && !same_storage(
                    &values[value.index() as usize],
                    &values[input.index() as usize],
                ))
        {
            return Some((value, index, copies));
        }
        copies.push(index);
        value = input;
    }
}

/// Price complete replacement sequences with the same overflow and missing-cost rules.
pub(super) fn operation_cycles<'a>(
    operations: impl IntoIterator<Item = &'a MidOperation>,
    values: &[MidValue],
) -> Option<u64> {
    operations.into_iter().try_fold(0u64, |sum, op| {
        crate::estimate::operation_cost(op, values)
            .map(|(cost, _, _)| sum.saturating_add(cost.total))
    })
}

pub(super) fn fusion_pays<'a>(
    fusion: &'static str,
    source: Option<OperationId>,
    before: impl IntoIterator<Item = &'a MidOperation>,
    after: impl IntoIterator<Item = &'a MidOperation>,
    values: &[MidValue],
) -> bool {
    let separate_cycles = operation_cycles(before, values);
    let fused_cycles = operation_cycles(after, values);
    let keep = separate_cycles
        .zip(fused_cycles)
        .is_some_and(|(a, b)| b < a);
    tracing::debug!(
        fusion,
        ?source,
        ?separate_cycles,
        ?fused_cycles,
        keep,
        "priced fusion"
    );
    keep
}
