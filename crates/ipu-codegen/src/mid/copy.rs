//! Coordinate-copy semantics and composition before tile expansion.

use crate::mid::MidOperationKind;
use crate::mid::{MidGraph, MidOperation, MidValue, MidValueId};
use crate::tensor::{AxisFactorView, TensorShape};
#[cfg(test)]
use ipu_target::Target;
use std::collections::{BTreeMap, BTreeSet};

/// Requested realization of a whole-device coordinate copy. Explicit requests
/// are checked by movement lowering; Automatic selects from the actual geometry.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum CopyPolicy {
    #[default]
    Automatic,
    /// Run a rearrangement kernel on corresponding resident shards.
    LocalKernel,
    /// Transfer compatible physical spans directly to their destination.
    DirectRetile,
    /// Gather complete native panels on relay tiles, then multicast them.
    /// Requires compatible physical order and complete, unpadded panels.
    GatherThenMulticast,
    /// Transfer logical values through row-major staging, then pack locally.
    StageLogicalThenTransform,
}

pub(crate) fn default_copy_policy(from: &crate::Layout, to: &crate::Layout) -> CopyPolicy {
    if from.order == to.order {
        CopyPolicy::DirectRetile
    } else {
        CopyPolicy::StageLogicalThenTransform
    }
}

/// Destination preparation for a selected copy. This is separate from its
/// logical/physical traversal policy; changing it does not change tensor values.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum PackingPolicy {
    /// Use direct word movement without destination packing scratch.
    Direct,
    /// For a format conversion, populate row-major scratch and pack locally.
    /// Copies already in their destination order need no packing scratch.
    #[default]
    Staged,
}

/// Map output coordinates back to the source: first add the window offsets,
/// then apply the optional factor-axis view. Layout/storage order is separate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoordinateMapping {
    pub offsets: Vec<u32>,
    pub view: Option<AxisFactorView>,
}

impl From<AxisFactorView> for CoordinateMapping {
    fn from(view: AxisFactorView) -> Self {
        Self {
            offsets: Vec::new(),
            view: Some(view),
        }
    }
}

impl CoordinateMapping {
    /// Omitted offsets and explicit zero offsets preserve the same coordinates.
    pub(crate) fn is_identity(&self) -> bool {
        self.view.is_none() && self.offsets.iter().all(|&offset| offset == 0)
    }

    /// Compose output -> intermediate -> source without materializing the
    /// intermediate. Return None when the result needs more than one factor
    /// view, or when an intermediate supplies logical zero padding.
    pub(crate) fn compose(
        &self,
        next: &Self,
        source: &TensorShape,
        intermediate: &TensorShape,
        output: &TensorShape,
    ) -> Option<Self> {
        fn fits(
            mapping: &CoordinateMapping,
            source: &TensorShape,
            output: &TensorShape,
        ) -> Option<()> {
            let shape = mapping
                .view
                .map_or_else(|| Some(source.clone()), |v| v.output_shape(source))?;
            if shape.0.len() != output.0.len() || mapping.offsets.len() > shape.0.len() {
                return None;
            }
            output
                .0
                .iter()
                .zip(&shape.0)
                .enumerate()
                .all(|(axis, (&size, &bound))| {
                    mapping
                        .offsets
                        .get(axis)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(size)
                        .is_some_and(|end| end <= bound)
                })
                .then_some(())
        }
        // Inverse factor moves can still absorb a following window. Combining
        // them with another view needs digit-order-aware composition.
        if next.view.is_some_and(|v| v.reversed)
            || (self.view.is_some_and(|v| v.reversed) && next.view.is_some())
        {
            return None;
        }
        fits(self, source, intermediate)?;
        let next_shape = next.view.map_or_else(
            || Some(intermediate.clone()),
            |v| v.output_shape(intermediate),
        )?;
        if output.0.len() != next_shape.0.len()
            || next.offsets.len() > output.0.len()
            || output.0.iter().enumerate().any(|(axis, &size)| {
                next.offsets
                    .get(axis)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(size)
                    .is_none()
            })
        {
            return None;
        }
        if fits(next, intermediate, output).is_none() {
            // The last copy may add padding if the removed copy did not crop
            // the source. Otherwise bypassing it could expose cropped values.
            let full = self
                .view
                .map_or_else(|| Some(source.clone()), |v| v.output_shape(source))?;
            if intermediate != &full
                || self.offsets.iter().any(|&offset| offset != 0)
                || output.0.len() != intermediate.0.len()
                || next.offsets.len() > output.0.len()
            {
                return None;
            }
        }
        let mut offsets = vec![0u32; output.0.len()];
        for (axis, offset) in offsets.iter_mut().enumerate() {
            *offset = self.offsets.get(axis).copied().unwrap_or(0);
        }
        let view = match (self.view, next.view) {
            (view, None) => view,
            (None, Some(view)) => {
                // A window before a view must retain the view's factor width.
                if source.0[view.split_axis] != intermediate.0[view.split_axis] {
                    return None;
                }
                offsets[view.merge_axis] = offsets[view.merge_axis].checked_mul(view.factor)?;
                Some(view)
            }
            (Some(first), Some(second)) => {
                if first.split_axis != second.split_axis
                    || first.merge_axis != second.merge_axis
                    || offsets[first.split_axis] != 0
                    || offsets[first.merge_axis] != 0
                    || first.output_shape(source)?.0[first.split_axis]
                        != intermediate.0[first.split_axis]
                {
                    return None;
                }
                Some(AxisFactorView::new(
                    first.split_axis,
                    first.merge_axis,
                    first.factor.checked_mul(second.factor)?,
                ))
            }
        };
        for (axis, offset) in offsets.iter_mut().enumerate() {
            *offset = offset.checked_add(next.offsets.get(axis).copied().unwrap_or(0))?;
        }
        while offsets.last() == Some(&0) {
            offsets.pop();
        }
        Some(Self { offsets, view })
    }
}

impl MidGraph {
    pub(crate) fn compose_copies(&mut self) {
        compose_region(&mut self.operations, &self.values, &self.outputs);
    }
}

/// Batch adjacent independent copies. Their local preparations precede one
/// shared exchange. Diagnostic builds retain semantic checkpoint boundaries.
pub(crate) fn independent_copy_prefix(
    operations: &[MidOperation],
    checkpoints: bool,
    storage_groups: &[MidValueId],
) -> usize {
    let group = |id: &MidValueId| storage_groups[id.index() as usize];
    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let source = operations.first().and_then(|operation| operation.source);
    operations
        .iter()
        .take_while(|operation| {
            if (checkpoints && operation.source != source)
                || !matches!(operation.kind, MidOperationKind::Copy { .. })
                || operation
                    .inputs
                    .iter()
                    .any(|id| outputs.contains(&group(id)))
                || operation
                    .results
                    .iter()
                    .any(|id| inputs.contains(&group(id)) || outputs.contains(&group(id)))
            {
                return false;
            }
            inputs.extend(operation.inputs.iter().map(group));
            outputs.extend(operation.results.iter().map(group));
            true
        })
        .count()
}

// Direct packing and relay requests belong to the present source/destination
// pair. Preserve those boundaries until composition can prove them realizable.
fn mapping(operation: &MidOperation) -> Option<(CoordinateMapping, CopyPolicy)> {
    match &operation.kind {
        MidOperationKind::Copy {
            mapping,
            policy,
            packing: crate::PackingPolicy::Staged,
        } if !matches!(
            policy,
            CopyPolicy::LocalKernel | CopyPolicy::GatherThenMulticast
        ) =>
        {
            Some((mapping.clone(), *policy))
        }
        _ => None,
    }
}

pub(super) fn compose_region(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) {
    for op in &mut *operations {
        if let MidOperationKind::Repeat(repeat) = &mut op.kind {
            compose_region(&mut repeat.body.operations, values, &repeat.body.yields);
        }
    }
    compose(operations, values, required);
}

pub(super) fn compose(
    operations: &mut Vec<MidOperation>,
    values: &[MidValue],
    required: &[MidValueId],
) {
    let mut uses = vec![0usize; values.len()];
    for input in operations
        .iter()
        .flat_map(MidOperation::read_values)
        .chain(required)
    {
        uses[input.index() as usize] += 1;
    }
    let mut producers = BTreeMap::<MidValueId, usize>::new();
    let mut removed = BTreeSet::new();
    for index in 0..operations.len() {
        let Some((mut next, mut policy)) = mapping(&operations[index]) else {
            // In-place compute and loop-carried storage can overwrite an
            // earlier source. Do not move a materialization across them.
            producers.clear();
            continue;
        };
        let ([input], [output]) = (
            operations[index].inputs.as_slice(),
            operations[index].results.as_slice(),
        ) else {
            continue;
        };
        let output = *output;
        let mut input = *input;
        while uses[input.index() as usize] == 1 {
            let Some(&producer) = producers.get(&input) else {
                break;
            };
            let (previous, previous_policy) = mapping(&operations[producer]).unwrap();
            let source = operations[producer].inputs[0];
            let intermediate = &values[input.index() as usize].tensor_type;
            let destination = &values[output.index() as usize].tensor_type;
            // Copy composition never crosses arithmetic, including casts.
            if values[source.index() as usize].tensor_type.format.precision
                != destination.format.precision
            {
                break;
            }
            // Keep pack-once staging when the first conversion needs local
            // packing. Native-panel copies can multicast straight to consumers;
            // retaining an intermediate merely adds a receive/forward hop.
            let source_format = &values[source.index() as usize].tensor_type.format;
            let exchange_only = previous.view.is_none()
                && (source_format.layout.order == intermediate.format.layout.order
                    || source_format.supports_micro_panel_exchange(&intermediate.format));
            if destination.format.layout.tiling.replicas
                > intermediate.format.layout.tiling.replicas
                && !exchange_only
            {
                break;
            }
            let Some(composed) = previous.compose(
                &next,
                &values[source.index() as usize].tensor_type.shape,
                &intermediate.shape,
                &destination.shape,
            ) else {
                break;
            };
            // Retiling before/after a logical transformation can use the final
            // owners directly: it does not require a distinct staging tensor.
            // Keep the logical traversal request on the composed copy. A direct
            // physical traversal is provable here only with unchanged element
            // order and no factor-axis mapping; otherwise keep the boundary.
            let merged_policy = match (previous_policy, policy) {
                (CopyPolicy::StageLogicalThenTransform, _)
                | (_, CopyPolicy::StageLogicalThenTransform) => {
                    CopyPolicy::StageLogicalThenTransform
                }
                (CopyPolicy::Automatic, p) | (p, CopyPolicy::Automatic) => p,
                (a, b) if a == b => a,
                _ => break,
            };
            if merged_policy == CopyPolicy::DirectRetile
                && (composed.view.is_some()
                    || source_format.layout.order != destination.format.layout.order)
            {
                break;
            }
            policy = merged_policy;
            next = composed;
            input = source;
            removed.insert(producer);
        }
        if input != operations[index].inputs[0] {
            operations[index].inputs[0] = input;
            operations[index].kind = MidOperationKind::Copy {
                policy,
                packing: crate::PackingPolicy::Staged,
                mapping: next,
            };
        }
        producers.insert(output, index);
    }
    super::rewrite::apply_edits(operations, &removed, BTreeMap::new());
}

#[cfg(test)]
mod tests {
    use crate::graph::ValueId;

    use crate::tensor::{Layout, Precision, TensorFormat, TensorTiling, TensorType};

    use super::*;

    #[test]
    fn linear_identity_copies_do_not_pay_for_an_exchange() {
        let (operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.format.layout = Layout::logical_linear(3, 4);
        }
        let (price, _, _) =
            crate::estimate::operation_cost(Target::Ipu21, &operations[0], &values, 3).unwrap();
        assert_eq!(price.exchange, 0);
        assert!(super::super::rewrite::same_storage(&values[0], &values[1]));

        values[1].tensor_type.shape.0[0] += 1;
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));
        values[1].tensor_type = values[0].tensor_type.clone();
        values[1].owners = crate::tensor::OwnerMap::rotated(1);
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));

        for value in &mut values {
            value.owners = crate::tensor::OwnerMap::default();
            value.tensor_type.format.layout = Layout::row_sharded(1);
        }
        values[1].tensor_type.format.layout.tiling.axes[0].shard_padding_multiple = 8;
        assert!(!super::super::rewrite::same_storage(&values[0], &values[1]));
    }

    fn coordinate(
        mapping: &CoordinateMapping,
        source: &TensorShape,
        mut point: Vec<u32>,
    ) -> Vec<u32> {
        for (axis, offset) in mapping.offsets.iter().enumerate() {
            point[axis] += offset;
        }
        if let Some(view) = mapping.view {
            let width = source.0[view.split_axis] / view.factor;
            point[view.split_axis] += point[view.merge_axis] % view.factor * width;
            point[view.merge_axis] /= view.factor;
        }
        point
    }

    #[test]
    fn composed_windows_and_factor_views_preserve_coordinates() {
        let mut rng = fastrand::Rng::with_seed(0x636f6d706f7365);
        let mut accepted = [0; 4];
        for _ in 0..4000 {
            let source = TensorShape(vec![12, 12, 12]);
            let mut make = |shape: &TensorShape| {
                let split = rng.usize(0..3);
                let merge = (split + rng.usize(1..3)) % 3;
                let view = rng.bool().then(|| AxisFactorView::new(split, merge, 2));
                let mut output =
                    view.map_or_else(|| Some(shape.clone()), |view| view.output_shape(shape))?;
                let offsets = output
                    .0
                    .iter_mut()
                    .map(|size| {
                        let offset = rng.u32(0..=(*size).min(1));
                        *size -= offset;
                        offset
                    })
                    .collect();
                Some((CoordinateMapping { offsets, view }, output))
            };
            let Some((first, middle)) = make(&source) else {
                continue;
            };
            let Some((second, output)) = make(&middle) else {
                continue;
            };
            let Some(composed) = first.compose(&second, &source, &middle, &output) else {
                continue;
            };
            accepted[usize::from(first.view.is_some()) * 2 + usize::from(second.view.is_some())] +=
                1;
            for _ in 0..10 {
                let point = output
                    .0
                    .iter()
                    .map(|&size| rng.u32(0..size))
                    .collect::<Vec<_>>();
                assert_eq!(
                    coordinate(&composed, &source, point.clone()),
                    coordinate(&first, &source, coordinate(&second, &middle, point))
                );
            }
        }
        assert!(accepted.iter().all(|&count| count > 0), "{accepted:?}");
    }

    fn chain() -> (Vec<MidOperation>, Vec<MidValue>) {
        let values = (0..4)
            .map(|index| MidValue {
                id: MidValueId(index),
                owners: crate::tensor::OwnerMap::default(),
                tensor_type: TensorType {
                    shape: TensorShape(vec![4, 16]),
                    format: TensorFormat {
                        precision: Precision::F16,
                        layout: Layout::row_major(TensorTiling::replicated(1)),
                    },
                },
                origin: ValueId::from_index(0),
                storage_group: MidValueId(index),
            })
            .collect();
        let operations = (1..4)
            .map(|index| MidOperation {
                source: None,
                inputs: vec![MidValueId(index - 1)],
                results: vec![MidValueId(index)],
                kind: MidOperationKind::Copy {
                    policy: crate::CopyPolicy::Automatic,
                    packing: crate::PackingPolicy::Staged,
                    mapping: CoordinateMapping::default(),
                },
                operands: Vec::new(),
                output_aliases: Vec::new(),
                output_windows: Vec::new(),
            })
            .collect();
        (operations, values)
    }

    #[test]
    fn copy_chains_drop_temporaries_idempotently() {
        let (mut operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.format.precision = Precision::F32;
        }
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);
        assert_eq!(operations[0].results, [MidValueId(3)]);
        let once = operations.clone();
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations, once);
    }

    #[test]
    fn copy_composition_stops_at_numeric_casts() {
        let (mut operations, mut values) = chain();
        values[0].tensor_type.format.precision = Precision::F32;
        operations[0].kind = MidOperationKind::Cast {
            from: Precision::F32,
            to: Precision::F16,
        };
        operations[0].operands = vec![crate::OperandIndexing::Elementwise { result: 0 }];
        let cast = operations[0].clone();
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[0], cast);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);
    }

    #[test]
    fn copy_composition_preserves_requested_realizations() {
        for first in [
            CopyPolicy::Automatic,
            CopyPolicy::LocalKernel,
            CopyPolicy::DirectRetile,
            CopyPolicy::StageLogicalThenTransform,
        ] {
            for second in [
                CopyPolicy::Automatic,
                CopyPolicy::LocalKernel,
                CopyPolicy::DirectRetile,
                CopyPolicy::StageLogicalThenTransform,
            ] {
                let (mut operations, values) = chain();
                operations.truncate(2);
                for (op, choice) in operations.iter_mut().zip([first, second]) {
                    if let MidOperationKind::Copy { policy, .. } = &mut op.kind {
                        *policy = choice;
                    }
                }
                compose(&mut operations, &values, &[MidValueId(2)]);
                let compatible =
                    first != CopyPolicy::LocalKernel && second != CopyPolicy::LocalKernel;
                assert_eq!(operations.len(), if compatible { 1 } else { 2 });
                if compatible {
                    let expected =
                        if [first, second].contains(&CopyPolicy::StageLogicalThenTransform) {
                            CopyPolicy::StageLogicalThenTransform
                        } else if second == CopyPolicy::Automatic {
                            first
                        } else {
                            second
                        };
                    assert!(matches!(operations[0].kind,
                        MidOperationKind::Copy { policy, .. } if policy == expected));
                }
            }
        }
    }

    #[test]
    fn composition_preserves_cropped_zeros() {
        let identity = CoordinateMapping::default();
        let view = CoordinateMapping::from(AxisFactorView::new(2, 0, 3));
        let source = TensorShape(vec![1, 2, 12]);
        let middle = TensorShape(vec![3, 2, 4]);
        let padded = TensorShape(vec![3, 2, 8]);
        assert_eq!(
            view.compose(&identity, &source, &middle, &padded),
            Some(view.clone())
        );
        let cropped = TensorShape(vec![3, 2, 2]);
        assert!(
            view.compose(&identity, &source, &cropped, &padded)
                .is_none()
        );
    }

    #[test]
    fn native_fp8_panel_materialization_composes_into_replication() {
        let (mut operations, mut values) = chain();
        for value in &mut values {
            value.tensor_type.shape = TensorShape(vec![64, 16]);
            value.tensor_type.format.precision = crate::Precision::F8F143 { scale_exponent: -4 };
            value.tensor_type.format.layout.order =
                crate::ElementOrder::BlockMajor(crate::BlockMajorOrder::Matrix {
                    row_block: 64,
                    column_block: 16,
                });
        }
        values[0].tensor_type.format.layout.order =
            crate::ElementOrder::Amp(crate::AmpOrder::TransposedLeft);
        values[2].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        values[3].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);
        assert_eq!(operations[0].results, [MidValueId(3)]);
    }

    #[test]
    fn composition_preserves_shared_results_broadcast_staging_and_padding() {
        let (mut operations, values) = chain();
        compose(&mut operations, &values, &[MidValueId(1), MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);

        let (mut operations, mut values) = chain();
        values[2].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        values[3].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 1);
        assert_eq!(operations[0].inputs, [MidValueId(0)]);

        let (mut operations, mut values) = chain();
        values[0].tensor_type.format.layout.order =
            crate::ElementOrder::Amp(crate::AmpOrder::Output);
        values[2].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        values[3].tensor_type.format.layout.tiling = TensorTiling::replicated(2);
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);

        let (mut operations, mut values) = chain();
        values[1].tensor_type.shape.0[1] += 4;
        values[2].tensor_type.shape.0[1] += 4;
        values[3].tensor_type.shape.0[1] += 4;
        compose(&mut operations, &values, &[MidValueId(3)]);
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[1].inputs, [MidValueId(1)]);
    }
}
