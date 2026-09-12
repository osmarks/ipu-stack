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

/// Recognize an unfused, whole-value FP16-to-FP8 local conversion.
pub(super) fn fp8_cast(op: &MidOperation, values: &[MidValue]) -> Option<(MidValueId, MidValueId)> {
    let local = match &op.kind {
        MidOperationKind::Convert(plan) => plan.strategy == ConversionStrategy::LocalKernel,
        MidOperationKind::Primitive(Primitive::Compute {
            kernel:
                TileKernelSpec::Cast {
                    from: Precision::F16,
                    to: Precision::F8F143 { .. },
                },
            operands,
            product: None,
            output_aliases,
        }) => operands.len() == 1 && operands[0].0.is_empty() && output_aliases.is_empty(),
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
        let identity = match &op.kind {
            MidOperationKind::Convert(_) => true,
            MidOperationKind::Primitive(Primitive::Copy { mapping, .. }) => {
                *mapping == CoordinateMapping::default()
            }
            _ => false,
        };
        if !identity
            || op.inputs.len() != 1
            || values[value.index() as usize].tensor_type.shape
                != values[op.inputs[0].index() as usize].tensor_type.shape
            || values[value.index() as usize].tensor_type.format.precision
                != values[op.inputs[0].index() as usize]
                    .tensor_type
                    .format
                    .precision
            || (identity_only
                && !same_storage(
                    &values[value.index() as usize],
                    &values[op.inputs[0].index() as usize],
                ))
        {
            return Some((value, index, copies));
        }
        copies.push(index);
        value = op.inputs[0];
    }
}
