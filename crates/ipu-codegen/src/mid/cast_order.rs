//! Searchable cast motion on expanded mid graphs, including compound operators.

use crate::graph::OperationId;
use crate::kernel::TileKernelSpec;
use crate::mid::{
    Compute, CoordinateMapping, MidOperation, MidOperationKind, MidProgram, MidValue, MidValueId,
    OperandIndexing,
};
use crate::tensor::{
    AmpOrder, AxisTiling, BlockMajorOrder, ElementOrder, Layout, Padding, Precision, TensorAxis,
    TensorFormat, TensorType,
};
use std::collections::{BTreeMap, BTreeSet};

/// Keep producer ownership while completing FP8's 32-element panels.
/// This rewrite preference avoids an intermediate F16 pack; it does not define
/// the set of representable tensor layouts or all legal kernel bindings.
pub(crate) fn producer_layout(input: &TensorType, target: &TensorFormat) -> Option<Layout> {
    let mut layout = cast_layout(input)?;
    if layout.order == ElementOrder::RowMajor && target.layout.order != ElementOrder::RowMajor {
        let resolved = layout.resolve(&input.shape).ok()?;
        let axes = resolved.axes()?;
        if axes.len() < 2 || !axes.last()?.extents_are_multiple_of(4) {
            return None;
        }
        if !axes.last()?.extents_are_multiple_of(32) {
            let rank = input.shape.0.len();
            if let Some(axis) = layout
                .tiling
                .axes
                .iter_mut()
                .find(|axis| axis.axis.resolve(rank) == Ok(rank - 1))
            {
                axis.shard_padding_multiple = axis.shard_padding_multiple.max(32);
                axis.padding = Padding::Zero;
            } else {
                layout.tiling.axes.push(
                    AxisTiling::new(TensorAxis::FromEnd(1), 1, 1, Padding::Zero)
                        .with_shard_padding_multiple(32),
                );
            }
        }
        layout.order = ElementOrder::Amp(AmpOrder::Left);
    }
    let quantized = TensorFormat {
        precision: target.precision,
        layout: layout.clone(),
    };
    layout.resolve(&input.shape).ok()?;
    (layout.order == target.layout.order || quantized.supports_micro_panel_exchange(target))
        .then_some(layout)
}

pub(crate) fn cast_layout(input: &TensorType) -> Option<Layout> {
    let mut layout = input.format.layout.clone();
    let axis_from_end = match layout.order {
        ElementOrder::Amp(AmpOrder::Left) => {
            let resolved = layout.resolve(&input.shape).ok()?;
            let columns = resolved.axes()?.last()?;
            if columns.extents_are_multiple_of(32) {
                return Some(layout);
            }
            // A narrow final tail is harmless, but padding every producer
            // panel would turn a bulk exchange into strided short packets.
            if !columns.complete_panels_except_tail(32) {
                return None;
            }
            let axis = layout.tiling.axes.iter_mut().find(|axis| {
                matches!(axis.axis, TensorAxis::FromEnd(1))
                    || axis.axis == TensorAxis::FromStart((input.shape.0.len() - 1) as u16)
            })?;
            axis.shard_padding_multiple = axis.shard_padding_multiple.max(32);
            return Some(layout);
        }
        ElementOrder::RowMajor
        | ElementOrder::Amp(AmpOrder::Output | AmpOrder::TransposedOutput) => {
            return Some(layout);
        }
        ElementOrder::Amp(AmpOrder::TransposedRight) => 1,
        ElementOrder::Amp(AmpOrder::TransposedLeft) => 2,
        ElementOrder::BlockMajor(
            BlockMajorOrder::Matrix { row_block, .. }
            | BlockMajorOrder::TransposedMatrix { row_block, .. },
        ) => return row_block.is_multiple_of(32).then_some(layout),
    };
    let valid = layout.resolve(&input.shape).ok().is_some_and(|resolved| {
        resolved.axes().is_some_and(|axes| {
            axes.len()
                .checked_sub(axis_from_end)
                .is_some_and(|axis| axes[axis].extents_are_multiple_of(32))
        })
    });
    valid.then_some(layout)
}

/// Ordinal among an operator's FP16 -> FP8 casts before any rewrites. Unlike
/// value IDs, this does not depend on allocations made for preceding operators.
pub(crate) type CastSite = (Option<OperationId>, u32);

impl MidProgram {
    pub(crate) fn reorder_casts(
        &mut self,
        selected: &BTreeSet<CastSite>,
        legacy: &BTreeSet<OperationId>,
    ) -> BTreeSet<CastSite> {
        let mut available = BTreeSet::new();
        reorder_region(
            &mut self.operations,
            &mut self.values,
            &self.outputs,
            selected,
            legacy,
            &mut BTreeMap::new(),
            &mut available,
        );
        available
    }
}

fn reorder_region(
    operations: &mut Vec<MidOperation>,
    values: &mut Vec<MidValue>,
    required: &[MidValueId],
    selected: &BTreeSet<CastSite>,
    legacy: &BTreeSet<OperationId>,
    ordinals: &mut BTreeMap<Option<OperationId>, u32>,
    available: &mut BTreeSet<CastSite>,
) {
    let producers = super::rewrite::single_use_producers(operations, required);
    let mut shared = Vec::<(MidValueId, MidValueId, usize)>::new();
    let mut removed = BTreeSet::new();
    let mut before = BTreeMap::<usize, Vec<MidOperation>>::new();
    for index in 0..operations.len() {
        if let MidOperationKind::Repeat(repeat) = &mut operations[index].kind {
            reorder_region(
                &mut repeat.body.operations,
                values,
                &repeat.body.yields,
                selected,
                legacy,
                ordinals,
                available,
            );
            continue;
        }
        let cast = &operations[index];
        let Some((mut input, output)) = super::rewrite::fp8_cast(cast, values) else {
            continue;
        };
        let ordinal = ordinals.entry(cast.source).or_default();
        let site = (cast.source, *ordinal);
        *ordinal += 1;
        let mut chain = Vec::new();
        let mut best = None;
        while let Some(&previous) = producers.get(&input) {
            let copy = &operations[previous];
            let Some(source) = super::rewrite::coordinate_copy_source(copy) else {
                break;
            };
            if previous >= index || copy.results != [input] {
                break;
            }
            let from = &values[source.index() as usize];
            let to = &values[input.index() as usize];
            if from.tensor_type.shape.0.len() != to.tensor_type.shape.0.len()
                || from.tensor_type.format.precision != Precision::F16
                || to.tensor_type.format.precision != Precision::F16
            {
                break;
            }
            chain.push(previous);
            input = source;
            // Cropping followed by padding must not resurrect discarded data.
            let final_shape = &values[output.index() as usize].tensor_type.shape.0;
            if chain.iter().any(|&i| {
                values[operations[i].results[0].index() as usize]
                    .tensor_type
                    .shape
                    .0
                    .iter()
                    .zip(&from.tensor_type.shape.0)
                    .zip(final_shape)
                    .any(|((extent, source), target)| *extent < (*source).min(*target))
            }) {
                continue;
            }
            let target = &values[output.index() as usize].tensor_type.format;
            let Some(layout) = producer_layout(&from.tensor_type, target) else {
                continue;
            };
            let format = TensorFormat {
                precision: target.precision,
                layout,
            };
            // Aliasing can be established by an earlier operation. Treat
            // in-place compute and nested regions as barriers rather than
            // inferring physical aliasing from Repeat storage groups.
            if operations[previous + 1..index]
                .iter()
                .enumerate()
                .any(|(offset, op)| {
                    !chain.contains(&(previous + 1 + offset)) && may_write_existing_storage(op)
                })
            {
                continue;
            }
            best = Some((source, format, chain.clone()));
        }
        let Some((input, format, chain)) = best else {
            tracing::debug!(?site, ?input, ?chain, tensor = ?values[input.index() as usize].tensor_type, "no early cast layout");
            continue;
        };
        tracing::debug!(?site, ?input, ?format, "available mid cast motion");
        available.insert(site);
        if !selected.contains(&site) && !cast.source.is_some_and(|id| legacy.contains(&id)) {
            continue;
        }
        let at = *chain.last().unwrap();
        let existing = shared
            .iter()
            .find(|&&(source, id, start)| {
                start < at
                    && source == input
                    && values[id.index() as usize].tensor_type.format == format
                    && !operations[start + 1..at]
                        .iter()
                        .any(may_write_existing_storage)
            })
            .map(|&(_, id, _)| id);
        let id = existing.unwrap_or_else(|| {
            let id = MidValueId(values.len() as u32);
            let mut value = values[input.index() as usize].clone();
            value.id = id;
            value.storage_group = id;
            value.tensor_type.format = format;
            values.push(value);
            let early = MidOperation {
                source: cast.source,
                inputs: vec![input],
                results: vec![id],
                kind: MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::Cast {
                        from: Precision::F16,
                        to: values[id.index() as usize].tensor_type.format.precision,
                    },
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: vec![],
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            };
            before.entry(at).or_default().push(early);
            shared.push((input, id, at));
            id
        });
        let mut copy = cast.clone();
        copy.inputs = vec![id];
        copy.kind = MidOperationKind::Copy {
            policy: crate::CopyPolicy::Automatic,
            packing: crate::PackingPolicy::Automatic,
            mapping: CoordinateMapping::default(),
            reuse_local: false,
        };
        copy.estimated_cycles = 0;
        copy.estimated_exchange_cycles = 0;
        before.entry(index).or_default().push(copy);
        removed.extend(chain);
        removed.insert(index);
    }
    super::rewrite::apply_edits(operations, &removed, before);
}

fn may_write_existing_storage(op: &MidOperation) -> bool {
    match &op.kind {
        MidOperationKind::Compute(compute) => {
            let output_aliases = compute.output_aliases();
            !output_aliases.is_empty()
        }
        MidOperationKind::Repeat(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate::estimate::MemoryPeaks;
    use crate::graph::{GraphInputKind, ValueId};
    use crate::low::CopyPolicy;
    use crate::mid::{MidInput, MidRegion, MidRepeat, rewrite};
    use crate::tensor::TensorTiling;

    use super::*;

    fn fixture() -> MidProgram {
        let mut layout = Layout::amp_left(1, 64);
        let input = TensorType::new([8, 64], Precision::F16, layout.clone());
        layout.tiling = TensorTiling::replicated(4);
        let staging = TensorType::new([8, 64], Precision::F16, layout.clone());
        let output = TensorType::new([8, 64], Precision::F8F143 { scale_exponent: -4 }, layout);
        let values = [input, staging, output]
            .into_iter()
            .enumerate()
            .map(|(i, tensor_type)| {
                let id = MidValueId(i as u32);
                MidValue {
                    id,
                    tile_offset: 0,
                    tensor_type,
                    origin: ValueId::from_index(0),
                    storage_group: MidValueId(0),
                }
            })
            .collect();
        let kinds = [
            MidOperationKind::Copy {
                mapping: CoordinateMapping::default(),
                reuse_local: false,
                policy: CopyPolicy::Automatic,
                packing: crate::PackingPolicy::Automatic,
            },
            MidOperationKind::Compute(Compute::cast(
                Precision::F16,
                Precision::F8F143 { scale_exponent: -4 },
            )),
        ];
        MidProgram {
            tile_count: 4,
            inputs: vec![MidInput {
                name: "input".into(),
                kind: GraphInputKind::Host,
                value: MidValueId(0),
            }],
            values,
            operations: kinds
                .into_iter()
                .enumerate()
                .map(|(i, kind)| MidOperation {
                    source: None,
                    inputs: vec![MidValueId(i as u32)],
                    results: vec![MidValueId(i as u32 + 1)],
                    kind,
                    estimated_cycles: 0,
                    estimated_exchange_cycles: 0,
                })
                .collect(),
            outputs: vec![MidValueId(2)],
            ..MidProgram::default()
        }
    }

    #[test]
    fn motion_preserves_escaping_values_and_does_not_cross_in_place_writes() {
        let mut mid = fixture();
        let intermediate = mid.operations[0].results[0];
        mid.outputs.push(intermediate);
        assert!(
            mid.reorder_casts(&BTreeSet::new(), &BTreeSet::new())
                .is_empty()
        );
        mid.outputs.pop();
        let source = mid.operations[0].inputs[0];
        let mut alias = mid.values[source.index() as usize].clone();
        alias.id = MidValueId(mid.values.len() as u32);
        mid.operations.insert(
            1,
            MidOperation {
                source: None,
                inputs: vec![source],
                results: vec![alias.id],
                kind: MidOperationKind::Compute(Compute::Kernel {
                    kernel: TileKernelSpec::Gelu,
                    operands: vec![OperandIndexing::Elementwise { result: 0 }],
                    output_aliases: vec![(0, 0)],
                }),
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            },
        );
        mid.values.push(alias);
        assert!(
            mid.reorder_casts(&BTreeSet::new(), &BTreeSet::new())
                .is_empty()
        );
    }

    #[test]
    fn repeat_choices_stay_inside_the_body_and_keep_bindings() {
        let mut mid = fixture();
        let yields = mid.outputs.clone();
        mid.operations = vec![MidOperation {
            source: None,
            inputs: vec![MidValueId(0)],
            results: yields.clone(),
            kind: MidOperationKind::Repeat(MidRepeat {
                count: 2,
                carried_inputs: 1,
                invariant_inputs: 0,
                iterated_inputs: vec![],
                body: MidRegion {
                    arguments: vec![MidValueId(0)],
                    operations: std::mem::take(&mut mid.operations),
                    yields: yields.clone(),
                    estimated_cycles: 0,
                    peak_memory: MemoryPeaks::default(),
                },
            }),
            estimated_cycles: 0,
            estimated_exchange_cycles: 0,
        }];
        let sites = mid.reorder_casts(&BTreeSet::new(), &BTreeSet::new());
        assert_eq!(sites.len(), 1);
        mid.reorder_casts(&sites, &BTreeSet::new());
        let MidOperationKind::Repeat(repeat) = &mid.operations[0].kind else {
            panic!()
        };
        assert_eq!(repeat.body.yields, yields);
        assert_eq!(repeat.body.arguments, [MidValueId(0)]);
        assert!(rewrite::fp8_cast(&repeat.body.operations[0], &mid.values).is_some());
        assert_eq!(mid.operations.len(), 1);
    }
    #[test]
    fn cast_motion_sees_staging_before_copy_composition_erases_it() {
        let mut mid = fixture();
        let source = mid.operations[0].inputs[0];
        let mut original = mid.values[source.index() as usize].clone();
        original.id = MidValueId(mid.values.len() as u32);
        original.storage_group = original.id;
        original.tensor_type.format.layout = Layout::logical_linear(4, 4);
        mid.operations.insert(
            0,
            MidOperation {
                source: None,
                inputs: vec![original.id],
                results: vec![source],
                kind: MidOperationKind::Copy {
                    policy: crate::CopyPolicy::Automatic,
                    packing: crate::PackingPolicy::Automatic,
                    mapping: CoordinateMapping::default(),
                    reuse_local: false,
                },
                estimated_cycles: 0,
                estimated_exchange_cycles: 0,
            },
        );
        mid.values.push(original);
        let mut rewritten = mid;
        let sites = rewritten.reorder_casts(&BTreeSet::new(), &BTreeSet::new());
        assert_eq!(sites.len(), 1);
        rewritten.reorder_casts(&sites, &BTreeSet::new());
        rewritten.compose_copies();
        let cast = rewritten
            .operations
            .iter()
            .find(|op| rewrite::fp8_cast(op, &rewritten.values).is_some())
            .unwrap();
        assert_eq!(cast.inputs, [source]);
        assert_eq!(rewritten.operations.len(), 3);
    }
}
